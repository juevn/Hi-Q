from json import JSONDecoder
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import os
from openai import OpenAI
import json
import tiktoken

MAX_CONTEXT_TOKENS = 120000


class LLMInference:
    def __init__(
        self,
        global_config,
    ):
        self.llm_model_name = global_config.model.llmModelName
        if self.llm_model_name == "llama":
            self.llama_sampling_params = {
                "temperature": global_config.model.temperature,
                "top_p": global_config.model.top_p,
                "max_new_tokens": global_config.model.max_tokens,
                "stop": ["\nInput:", "Input:"],
            }
            self.llm_addr = "http://localhost:30000/generate"
        elif "gpt" in self.llm_model_name:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
            self.llm_hyperparams = {
                "temperature": global_config.model.temperature,
                "top_p": global_config.model.top_p,
                "max_tokens": global_config.model.max_tokens,
            }

    def get_tokenizer(self, model_name: str):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str, tokenizer) -> int:
        return len(tokenizer.encode(text))

    def truncate_user_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        max_tokens: int,
        margin: int = 200,  # response + safety buffer
    ):
        tokenizer = self.get_tokenizer(model_name)

        system_tokens = self.count_tokens(system_prompt, tokenizer)
        max_user_tokens = max_tokens - system_tokens - margin

        if max_user_tokens <= 0:
            raise ValueError("System prompt alone exceeds token limit.")

        user_tokens = tokenizer.encode(user_prompt)

        if len(user_tokens) <= max_user_tokens:
            return user_prompt

        truncated_tokens = user_tokens[-max_user_tokens:]
        return tokenizer.decode(truncated_tokens)

    def extract_json_objects(self, text, decoder=JSONDecoder()):
        """Find JSON objects in text, and yield the decoded JSON data"""
        stack = []
        start_idx = None
        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start_idx = i
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        json_str = text[start_idx : i + 1]
                        try:
                            return decoder.decode(json_str)
                        except Exception:
                            return None
        return None

    def validate(self, obj: Any, schema: Dict[str, Any]) -> None:
        """
        schema DSL examples:
          {"type": dict, "required": {key: <subschema>, ...}}
          {"type": list, "items": <subschema>}
          {"type": str/int/float/bool}
          {"enum": ["yes","no"]} etc.
        if fail, raise ValueError
        """
        if "enum" in schema:
            if obj not in schema["enum"]:
                raise ValueError(
                    f"enum mismatch: got={obj}, expected one of {schema['enum']}"
                )
            return

        expected_type = schema.get("type", None)
        if expected_type is not None and not isinstance(obj, expected_type):
            raise ValueError(
                f"type mismatch: got={type(obj)}, expected={expected_type}"
            )

        if expected_type is dict:
            required = schema.get("required", {})
            for k, subschema in required.items():
                if k not in obj:
                    raise ValueError(f"missing key: {k}")
                self.validate(obj[k], subschema)

        if expected_type is list:
            items_schema = schema.get("items", None)
            if items_schema is not None:
                for idx, item in enumerate(obj):
                    try:
                        self.validate(item, items_schema)
                    except ValueError as e:
                        raise ValueError(f"invalid list item at {idx}: {e}")

    def inference(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if "gpt" in self.llm_model_name:
            return self.inference_gpt(
                system_prompt, user_prompt, schema, sampling_overrides
            )
        elif self.llm_model_name == "llama":
            return self.inference_llama(
                system_prompt, user_prompt, schema, sampling_overrides
            )
        else:
            raise ValueError(f"Unknown LLM model name: {self.llm_model_name}")

    def inference_gpt(
        self,
        system_prompt: str,
        user_prompt: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:

        safe_user_prompt = self.truncate_user_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=self.llm_model_name,
            max_tokens=MAX_CONTEXT_TOKENS,
        )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": safe_user_prompt},
        ]
        hyperparams = dict(self.llm_hyperparams)
        if sampling_overrides:
            for key, value in sampling_overrides.items():
                if key in hyperparams and value is not None:
                    hyperparams[key] = value
        completion = (
            self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=prompt,
                **hyperparams,
            )
            .choices[0]
            .message.content
        )

        try:
            response = json.loads(completion)
        except json.JSONDecodeError:
            response = self.extract_json_objects(completion)

        return response

    def inference_llama(
        self,
        system_prompt: str,
        user_prompt: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        prompt_text = f"{system_prompt}\n\n{user_prompt}"

        output = []
        try:
            sampling_params = dict(self.llama_sampling_params)
            if sampling_overrides:
                for key, value in sampling_overrides.items():
                    if key in sampling_params and value is not None:
                        sampling_params[key] = value
            resp = requests.post(
                self.llm_addr,
                json={
                    "text": [prompt_text],
                    "sampling_params": sampling_params,
                },
                timeout=180,
            )
            resp.raise_for_status()
            qa_response = resp.json()
            json_object = self.extract_json_objects(qa_response[0]["text"])
            output = json_object
        except Exception as e:
            qa_response = ("", {"error": str(e)}, False)

        return output
