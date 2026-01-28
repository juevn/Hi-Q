# Hi-Q

Experimental code for multi-hop QA that combines **Dense Retrieval + LLM reasoning**.  
The default flow is:

- Try a **single-hop** answer first.
- If that fails, expand to **multi-hop retrieval/reasoning (with beam search)**.

Supported datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue

---

## Project Structure

- `src/model/hi_q/hi_q.py` : main entry script (Hydra-based)
- `src/model/hi_q/llm.py` : LLM client (OpenAI or local Llama server)
- `src/model/hi_q/embedding_model/NVEmbedV2.py` : NV-Embed-v2 embedder
- `src/model/hi_q/retriever.py` : Dense retriever + cache
- `src/dataset/*` : dataset loaders
- `config/` : Hydra configs (model/benchmark)
- `dataset/` : sample dataset/corpus JSON
- `outputs/` : results output directory
- `data/cache/` : retrieval index cache (auto-generated)

---

## Setup

### 1) Path configuration

Replace `{YOUR_ROOT_DIR}` in `config/model/baseline.yaml` with your repo path.
Example: `/home/jekim/Hi-Q`

```yaml
# config/model/baseline.yaml
paths:
  root_dir: /home/jekim/Hi-Q
```

It is also recommended to set `root_dir_path` in `config/config.yaml` to the same value (for clarity).

### 2) Install dependencies

There is no requirements file, so install dependencies manually.
Minimal set:

```bash
pip install torch transformers accelerate hydra-core omegaconf openai tiktoken requests tqdm numpy
```

If you use a GPU, install a CUDA-matched version of torch.

### 3) LLM configuration

- Default is an OpenAI model (`gpt-4o-mini`).
- For OpenAI, set:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

- For local Llama:
  - Set `llmModelName: llama` in `config/model/baseline.yaml`
  - Ensure `http://localhost:30000/generate` is running

---

## Running

Main entrypoint:

```bash
python /home/jekim/Hi-Q/src/model/hi_q/hi_q.py
```

**Important:** the repo root must be in `PYTHONPATH` for `src.*` imports.  
Recommended:

```bash
PYTHONPATH=/home/jekim/Hi-Q python /home/jekim/Hi-Q/src/model/hi_q/hi_q.py
```

### Switch dataset

Use Hydra override:

```bash
PYTHONPATH=/home/jekim/Hi-Q python /home/jekim/Hi-Q/src/model/hi_q/hi_q.py benchmark=hotpotqa
```

Supported benchmarks:
- `hotpotqa`
- `2wikimultihopqa`
- `musique` (default)

### Multi-GPU / distributed

Using Accelerate:

```bash
accelerate launch /home/jekim/Hi-Q/src/model/hi_q/hi_q.py
```

---

## Outputs

Results are saved to:

- `outputs/<benchmark>/qa_results_<rank>.json` (per-process JSONL)
- `outputs/<benchmark>/retrieved_results_<rank>.json`
- `outputs/<benchmark>/qa_results.json` (merged on main process)
- `outputs/<benchmark>/retrieved_results.json`

Retrieval embedding cache:

```
data/cache/<benchmark>/passage_embeddings.npz
```

---

## Notes / Troubleshooting

- In `src/model/hi_q/retriever.py`, the `NVEmbedV2Embedder` import path is
  `src.model.baseline...`, which can cause `ModuleNotFoundError`.  
  Change it to `src.model.hi_q.embedding_model.NVEmbedV2` if needed.
- Dataset files are included in `dataset/`.
- Wrong paths will raise `FileNotFoundError`.
