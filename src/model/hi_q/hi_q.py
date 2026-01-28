from http import client
from typing import List, Dict, Any, Optional, Tuple
import json
import random
from json import JSONDecoder
import requests
from torch import topk
from tqdm import tqdm

from src.model.hi_q.prompt import (
    dq_prompt,
    singlehop_prompt,
    subquery_prompt,
    final_answer_prompt,
    repair_prompt,
)
from dataclasses import dataclass, field

from src.dataset.dataloading import load_dataset
import hydra
from omegaconf import DictConfig
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME

from src.model.hi_q.embedding_model.NVEmbedV2 import NVEmbedV2Embedder
from src.model.hi_q.retriever import DenseRetriever
from src.model.hi_q.llm import LLMInference

from accelerate import Accelerator
from accelerate.utils import gather_object
import os

import logging

logger = logging.getLogger(__name__)


@dataclass
class Need:
    id: str
    text: str
    depends_on: List[str] = field(default_factory=list)
    subquery: str = ""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Need":
        return Need(
            id=data["id"],
            text=data["text"],
            depends_on=data.get("depends_on", []),
            subquery=data["subquery"],
        )


@dataclass
class RecursiveContext:
    history_answers: Dict[str, Any] = field(default_factory=dict)
    history_order: List[str] = field(default_factory=list)
    resolved_answers: Dict[str, Any] = field(default_factory=dict)
    resolve_order: List[str] = field(default_factory=list)
    retrieved_docs: List[Tuple[str, float]] = field(default_factory=list)
    expansions: int = 0
    max_depth: int = 4
    max_expansions: int = 20


class HiQ:

    def __init__(self, global_config, device):
        self.global_config = global_config
        self.llm_model_name = global_config.model.llmModelName
        self.embedding_model_name = global_config.model.embeddingName
        self.embedder = NVEmbedV2Embedder(global_config, device=device)
        self.retriever = DenseRetriever(self.embedder)
        self.retrieve_topk = global_config.model.get("retrieve_top_k", 5)
        # Retriever Setup
        with open(global_config.benchmark.corpus_path, "r") as f:
            corpus_passages = json.load(f)
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus_passages]
        self.retriever.build_or_load_index(
            passages=docs,
            cache_path=global_config.model.retrieve_cache_path,
        )
        # LLM Setup
        self.llm = LLMInference(global_config)
        # Beam Search Setup
        self.beam_width = global_config.model.get("beam_width", 3)
        self.confidence_threshold = global_config.model.get("confidence_threshold", 0.7)

    def is_valid_answer(self, a: Any) -> bool:
        if isinstance(a, str):
            s = a.strip().lower()
            if not s:
                return False
            if s in {"null", "none"}:
                return False
            return True
        if isinstance(a, List):
            return len(a) > 0

    def eval_recall(self, gold_docs, retrieved_docs):
        relevant_retrieved = set(retrieved_docs) & set(gold_docs)
        return len(relevant_retrieved) / len(set(gold_docs))

    def run(self, query: str, gold_docs: List[str] = []):
        initial_answer, _, _, initial_docs = self._single_hop(query)
        if self.is_valid_answer(initial_answer):
            recall_k = self.eval_recall(gold_docs, initial_docs)  # top-5
            result_entry: Dict[str, Any] = {
                "original_query": query,
                "pred_answer": initial_answer,
                "beam_history": None,
                "beam_graph": None,
                "hop": "single-hop",
                "recall@k": recall_k,
            }
            retrieved_docs = initial_docs
        else:
            answer, beam_history, beam_graph, retrieved_docs = self._multi_hop(query)
            recall_k = self.eval_recall(gold_docs, retrieved_docs)
            result_entry: Dict[str, Any] = {
                "original_query": query,
                "pred_answer": answer,
                "beam_history": beam_history,
                "beam_graph": beam_graph,
                "hop": "multi-hop",
                "recall@k": recall_k,
            }
        return result_entry, retrieved_docs

    def retrieve_topk_docs(
        self,
        query: str,
        input_topk: Optional[int] = None,
        return_score: bool = False,
    ):
        """Retrieve documents for a single query using the retriever model."""
        top_k = input_topk if input_topk else self.retrieve_topk
        results = self.retriever.search([query], top_k)[0]

        if return_score:
            return [(doc, score) for doc, score in results]
        else:
            return [doc for doc, _ in results]

    def single_hop_with_evidence_alignment(
        self,
        question,
        docs_for_question: List[str],
    ) -> str:
        return self.single_hop_reader(
            question=question,
            docs_for_question=docs_for_question,
        )

    def reader_with_evidence_alignment(
        self,
        question,
        docs_for_question: List[str],
        sub_question: str = "",
        docs: str = "",
    ) -> str:
        return self.reader(
            original_question=question,
            sub_question=sub_question,
            docs_for_question=docs_for_question,
            retrieved_docs=docs,
        )

    def single_hop_reader(
        self,
        question: str,
        docs_for_question: List[str],
    ) -> str:
        """Answer single-hop queries using the reader model."""
        docs_text = (
            "\n\n".join(docs_for_question)
            if isinstance(docs_for_question, list)
            else str(docs_for_question)
        )
        answer_json = self.llm.inference(
            singlehop_prompt.SINGLEHOP_SYSTEM_PROMPT,
            singlehop_prompt.SINGLEHOP_USER_PROMPT.format(
                question=question, documents=docs_text
            ),
        )
        answer = ""
        if isinstance(answer_json, dict):
            answer = answer_json.get("answer") or ""
        return answer

    def reader(
        self,
        original_question: str,
        sub_question: str,
        docs_for_question: List[str],
        retrieved_docs: str = "",
    ) -> str:
        """Answer single-hop queries using the reader model."""
        docs_text = (
            "\n\n".join(docs_for_question)
            if isinstance(docs_for_question, list)
            else str(docs_for_question)
        )
        answer_json = self.llm.inference(
            singlehop_prompt.READING_SYSTEM_PROMPT,
            singlehop_prompt.READING_USER_PROMPT.format(
                original_question=original_question,
                sub_question=sub_question,
                documents=docs_text,
                retrieved_docs=retrieved_docs,
            ),
        )
        answer = ""
        if isinstance(answer_json, dict):
            answer = answer_json.get("answer") or ""
        return answer

    def decompose_query_into_two_needs(self, query: str) -> List[Need]:
        """Decompose a single query into its needs."""
        json_object = self.llm.inference(
            dq_prompt.DECOMPOSE_QUESTION_TWO_SYSTEM_PROMPT,
            dq_prompt.DECOMPOSE_QUESTION_TWO_USER_PROMPT.format(question=query),
        )
        if json_object is None:
            return []
        need_objects = self.check_format_of_decomposition(json_object)
        if need_objects:  # valid decomposition
            new_json_object = self.repair_decomposition(query, json_object)
            if new_json_object:
                new_need_objects = self.check_format_of_decomposition(new_json_object)
                return new_need_objects
            else:
                return need_objects
        else:  # not decompose
            return []

    def check_format_of_decomposition(self, json_object: Dict[str, Any]):
        needs = json_object.get("needs", [])
        if not needs or len(needs) != 2:
            return []

        need_objects: List[Need] = []
        for n in needs:
            if isinstance(n, Need):
                need_objects.append(n)
            else:
                try:
                    need_objects.append(Need.from_dict(n))
                except Exception:
                    continue
        if len(need_objects) != 2:
            return []
        need_objects.sort(key=lambda x: len(x.depends_on))
        if (
            need_objects[0].id == "N1"
            and need_objects[1].id == "N2"
            and len(need_objects[0].depends_on) == 0
            and len(need_objects[1].depends_on) > 0
            and need_objects[1].depends_on[0] == need_objects[0].id
        ):
            return need_objects
        else:
            return []

    def repair_decomposition(self, question, json_object: Dict[str, Any]) -> List[Need]:
        if "thought" in json_object:
            del json_object["thought"]
        repair_response = self.llm.inference(
            repair_prompt.DECOMPOSITION_REPAIR_SYSTEM_PROMPT,
            repair_prompt.DECOMPOSITION_REPAIR_USER_PROMPT.format(
                question=question,
                decomposition=json.dumps(json_object),
            ),
        )
        if not repair_response or not repair_response.get("needs", []):
            return []
        else:
            return repair_response

    def _multi_hop(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Solve the query by recursively decomposing and refining needs."""
        retrieved_docs: List[str] = []
        need_objects = self.decompose_query_into_two_needs(query)
        if not need_objects:
            return self._single_hop_with_increasing_topk(query)

        need_map: Dict[str, Need] = {}
        for need in need_objects:
            need_map[need.id] = need

        context = RecursiveContext(
            max_depth=self.global_config.model.get("recursive_max_depth", 4),
            max_expansions=self.global_config.model.get("recursive_max_expansions", 20),
            retrieved_docs=retrieved_docs,
        )
        root_answers: Dict[str, str] = {}
        for root_need in [need_map["N1"], need_map["N2"]]:
            root_answers[root_need.id] = self._solve_need(
                question=query,
                need_id=root_need.id,
                need_map=need_map,
                context=context,
                depth=1,
            )

        need_map["root"] = Need(
            id="root",
            text=query,
            depends_on=["N1", "N2"],
            subquery=query,
        )
        self._refine_need(
            question=query,
            need=need_map["root"],
            context=context,
        )

        best_scores: Dict[str, float] = {}
        for doc, score in context.retrieved_docs:
            doc_key = doc if isinstance(doc, str) else str(doc)
            prev_score = best_scores.get(doc_key)
            if prev_score is None or score > prev_score:
                best_scores[doc_key] = score
        if best_scores:
            final_retrieved_docs = [
                doc
                for doc, _ in sorted(
                    best_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]

        if not context.resolved_answers:
            history_text = self._build_history_text(
                need_map=need_map,
                resolve_order=context.history_order,
                resolved_answers=context.history_answers,
            )
            return None, history_text, None, final_retrieved_docs

        history_text = self._build_history_text(
            need_map=need_map,
            resolve_order=context.resolve_order,
            resolved_answers=context.resolved_answers,
        )
        retrieved_docs = self._format_docs_as_context(context.retrieved_docs)
        final_resp = self.llm.inference(
            final_answer_prompt.FINAL_ANSWER_SYSTEM_PROMPT,
            final_answer_prompt.FINAL_ANSWER_USER_PROMPT.format(
                question=query,
                history=history_text,
                docs=retrieved_docs,
            ),
        )

        history_text = self._build_history_text(
            need_map=need_map,
            resolve_order=context.history_order,
            resolved_answers=context.history_answers,
        )
        if isinstance(final_resp, dict):
            return (
                final_resp.get("answer") or "",
                history_text,
                None,
                final_retrieved_docs,
            )
        return None, history_text, None, final_retrieved_docs

    def _single_hop_with_increasing_topk(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        for k in [10, 15, 20]:
            answer, _, _, docs = self._single_hop(query, input_topk=k)
            if answer:
                return answer, None, None, docs[:15]
        return None, None, None, docs[:15]

    def _single_hop(
        self, query: str, input_topk: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        docs = self.retrieve_topk_docs(query, input_topk=input_topk, return_score=False)
        answer = self.single_hop_with_evidence_alignment(
            question=query, docs_for_question=docs
        )
        if not answer:
            return None, None, None, docs
        return answer, None, None, docs

    def _reader_with_increasing_topk(
        self, question: str, sub_question: str, docs: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        for k in [10, 15, 20]:
            answer, _, _, docs_with_score = self._reader(
                question, sub_question, docs, input_topk=k
            )
            if answer:
                return answer, None, None, docs_with_score
        return None, None, None, docs_with_score

    def _reader(
        self,
        question: str,
        sub_question: str,
        docs: str,
        input_topk: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        docs_with_score = self.retrieve_topk_docs(
            sub_question,
            input_topk=input_topk,
            return_score=True,
        )
        docs_only = [doc for doc, _ in docs_with_score]
        answer = self.reader_with_evidence_alignment(
            question=question,
            docs_for_question=docs_only,
            sub_question=sub_question,
            docs=docs,
        )
        if not answer:
            return None, None, None, docs_with_score
        return answer, None, None, docs_with_score

    def _solve_need(
        self,
        question: str,
        need_id: str,
        need_map: Dict[str, Need],
        context: RecursiveContext,
        depth: int,
    ) -> Optional[str]:

        need = need_map.get(need_id)
        if need is None:
            return None
        answer = self._refine_need(
            question=question,
            need=need,
            context=context,
        )
        if answer:
            return answer

        if depth >= context.max_depth:
            return None
        if context.expansions >= context.max_expansions:
            return None

        child_needs = self._decompose_need(need, need_map, context)
        if not child_needs or len(child_needs) == 0:
            retrieved_docs = self._format_docs_as_context(context.retrieved_docs)
            answer, _, _, docs_with_score = self._reader_with_increasing_topk(
                question, need.subquery or "", docs=retrieved_docs
            )
            context.retrieved_docs.extend(docs_with_score)
            return answer

        child_answers: Dict[str, str] = {}
        for child in child_needs:
            child_answers[child.id] = self._solve_need(
                question=question,
                need_id=child.id,
                need_map=need_map,
                context=context,
                depth=depth + 1,
            )

        for child in child_needs:
            need.depends_on.append(child.id)
        answer = self._refine_need(
            question=question,
            need=need,
            context=context,
        )
        for child in child_needs:
            if child.id in need.depends_on:
                need.depends_on.remove(child.id)
        return answer

    def _refine_need(
        self,
        question: str,
        need: Need,
        context: RecursiveContext,
    ) -> Optional[str]:
        subquery = (need.subquery or "").strip()
        if not subquery:
            return None

        if need.depends_on:
            plan_resp = self.llm.inference(
                system_prompt=subquery_prompt.SUBQUERY_SYSTEM_PROMPT,
                user_prompt=subquery_prompt.SUBQUERY_USER_PROMPT.format(
                    question=question,
                    need_subquery=subquery,
                    docs=self._format_docs_as_context(context.retrieved_docs),
                ),
            )
            planned = plan_resp.get("query") if isinstance(plan_resp, dict) else None
            subquery = (planned or subquery).strip()
            need.subquery = subquery

        retrieved_docs = self._format_docs_as_context(context.retrieved_docs)
        answer, _, _, docs_for_need_with_score = self._reader(
            question, subquery, retrieved_docs
        )
        context.retrieved_docs.extend(docs_for_need_with_score)
        context.history_answers[need.id] = answer  # for history
        context.history_order.append(need.id)  # for history

        if not self.is_valid_answer(answer):
            return None

        context.resolved_answers[need.id] = answer
        context.resolve_order.append(need.id)
        return answer

    def _decompose_need(
        self,
        need: Need,
        need_map: Dict[str, Need],
        context: RecursiveContext,
    ) -> List[Need]:
        seed_query = (need.subquery or "").strip()
        if not seed_query:
            return []

        context.expansions += 1
        children_raw = self.decompose_query_into_two_needs(seed_query)
        if not children_raw:
            return []
        if len(children_raw) != 2:
            return []

        children: List[Need] = []
        for idx, child in enumerate(children_raw):
            if isinstance(child, Need):
                child.id = f"{need.id}_{idx + 1}"
                if len(child.depends_on) != 0:
                    refined_depends_on = [f"{need.id}_1"]
                    child.depends_on = list(set(refined_depends_on + need.depends_on))
            else:
                try:
                    child = Need.from_dict(child)
                    child.id = f"{need.id}_{idx + 1}"
                    if len(child.depends_on) != 0:
                        refined_depends_on = [f"{need.id}_1"]
                        child.depends_on = list(
                            set(refined_depends_on + need.depends_on)
                        )
                except Exception:
                    continue
            children.append(child)
            need_map[child.id] = child
        return children

    def _build_history_text(
        self,
        need_map: Dict[str, Need],
        resolve_order: List[str],
        resolved_answers: Dict[str, Any],
    ) -> str:
        history_lines: List[str] = []
        for nid in resolve_order:
            need_obj = need_map.get(nid)
            need_subquery = need_obj.subquery if need_obj is not None else nid
            ans = resolved_answers.get(nid, "None")
            if isinstance(ans, list):
                ans_str = ", ".join(map(str, ans))
            else:
                ans_str = str(ans)
            history_lines.append(f"- {nid}: {need_subquery} => {ans_str}")
        return "\n".join(history_lines) if history_lines else "(empty)"

    def _format_docs_as_context(self, docs) -> str:
        if not docs:
            return "(empty)"
        selected = docs
        blocks: List[str] = []
        seen: set = set()
        for doc in selected:
            doc_item = doc[0] if isinstance(doc, (tuple, list)) and doc else doc
            doc_str = str(doc_item)
            if "\n" in doc_str:
                raw_title, raw_content = doc_str.split("\n", 1)
            else:
                raw_title, raw_content = "", doc_str
            title = raw_title.strip()
            content = raw_content.strip()
            doc_key = f"{title}\n{content}"
            if doc_key in seen:
                continue
            seen.add(doc_key)
            display_index = len(blocks) + 1
            display_title = (title or f"Document {display_index}").strip()
            blocks.append(
                f"id: D{display_index}\n"
                f"title: {display_title}\n"
                f"content: {content}"
            )
        return "\n\n".join(blocks)


def get_gold_docs(samples: List, dataset_name: str = None) -> Dict[str, List[str]]:
    gold_docs: Dict[str, List[str]] = {}
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name.startswith("hotpotqa"):
                gold_doc = [
                    item[0] + "\n" + "".join(item[1])
                    for item in gold_title_and_content_list
                ]
            else:
                gold_doc = [
                    item[0] + "\n" + " ".join(item[1])
                    for item in gold_title_and_content_list
                ]
        elif "contexts" in sample:
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            # Musique
            assert (
                "paragraphs" in sample
            ), "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                item["title"]
                + "\n"
                + (item["text"] if "text" in item else item["paragraph_text"])
                for item in gold_paragraphs
            ]

        gold_doc = list(set(gold_doc))
        query_id = sample.get("id")
        if query_id is None:
            raise KeyError("sample is missing `id` needed for gold_docs mapping")
        gold_docs[query_id] = gold_doc
    return gold_docs


def splitDataset(dataset, acc):  # Interleave sharding
    idxs = range(len(dataset))
    shard_idxs = idxs[acc.process_index :: acc.num_processes]
    return [dataset[i] for i in tqdm(shard_idxs, disable=not acc.is_local_main_process)]


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    acc = Accelerator()
    device = acc.device

    outputPath = os.path.join(
        cfg.model.output_path, f"qa_results_{acc.process_index}.json"
    )
    retrieved_output_path = os.path.join(
        cfg.model.output_path, f"retrieved_results_{acc.process_index}.json"
    )
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    dataset = load_dataset(cfg)
    dataset.data = splitDataset(dataset.data, acc)
    gold_docs = get_gold_docs(dataset.data, dataset_name=cfg.benchmark.name)
    model = HiQ(global_config=cfg, device=device)
    logging.basicConfig(level=logging.INFO)

    results: List[Dict[str, Any]] = []
    retrieved_results: List[Dict[str, Any]] = []
    for sample in tqdm(dataset.data, desc="Processing queries"):
        query = sample.get("question")
        gold_docs_for_sample = gold_docs.get(sample.get("id"), [])
        if not query:
            continue
        result, retrieved_docs = model.run(query=query, gold_docs=gold_docs_for_sample)
        result["gold_answer"] = sample.get("answer")
        result["id"] = sample.get("id")
        results.append(result)
        with open(outputPath, "a+") as f:
            json.dump(result, f)
            f.write("\n")

        retrieved_result = {}
        retrieved_result["id"] = sample.get("id")
        retrieved_result["retrieved_docs"] = retrieved_docs
        retrieved_results.append(retrieved_result)
        with open(retrieved_output_path, "a+") as f:
            json.dump(retrieved_result, f)
            f.write("\n")

    gathered = gather_object(results)
    retrieved_gathered = gather_object(retrieved_results)
    if acc.is_main_process:
        final_output_path = os.path.join(cfg.model.output_path, "qa_results.json")
        with open(final_output_path, "w") as f:
            json.dump(gathered, f, indent=4)
        final_retrieved_output_path = os.path.join(
            cfg.model.output_path, "retrieved_results.json"
        )
        with open(final_retrieved_output_path, "w") as f:
            json.dump(retrieved_gathered, f, indent=4)


if __name__ == "__main__":
    main()
