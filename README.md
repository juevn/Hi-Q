# Hi-Q

## Method Overview

This framework mitigates granularity mismatch via **hierarchical evidence-guided query refinement**.  
Starting from a coarse query, it **recursively decomposes** it into two finer sub-queries until each aligns with retrievable, atomic evidence.  
Retrieval success signals adequate alignment, while failure triggers further decomposition, forming a **binary decomposition tree** whose leaves represent the optimal granularity for evidence acquisition.  
To prevent semantic drift and error propagation, a **round-trip consistency check** ensures sub-queries can reconstruct the original intent.

<img src="image/ours_overview.png" alt="Method overview diagram" width="80%" />

---

## Project Structure

- `src/model/hi_q/hi_q.py` : main entry script (Hydra-based)
- `src/model/hi_q/llm.py` : LLM client (OpenAI or local Llama server)
- `src/model/hi_q/embedding_model/NVEmbedV2.py` : NV-Embed-v2 embedder
- `src/model/hi_q/retriever.py` : Dense retriever + cache
- `src/dataset/*` : dataset loaders
- `config/` : Hydra configs (model/benchmark)
- `dataset/` : dataset/corpus JSON
- `outputs/` : results output directory
- `data/cache/` : retrieval index cache (auto-generated)

---

## Setup

### 1) Path configuration

Replace `{YOUR_ROOT_DIR}` in `config/model/baseline.yaml` with your repo path.
Example: `/path/to/Hi-Q`

```yaml
# config/model/baseline.yaml
paths:
  root_dir: /path/to/Hi-Q
```

It is also recommended to set `root_dir_path` in `config/config.yaml` to the same value (for clarity).

### 2) LLM configuration

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
python /path/to/Hi-Q/src/model/hi_q/hi_q.py
```

**Important:** the repo root must be in `PYTHONPATH` for `src.*` imports.  
Recommended:

```bash
PYTHONPATH=/path/to/Hi-Q python /path/to/Hi-Q/src/model/hi_q/hi_q.py
```

### Switch dataset

Edit the config file instead of using CLI overrides.

In `config/config.yaml`, change the default benchmark:

```yaml
defaults:
  - _self_
  - model: baseline
  - benchmark: hotpotqa
```

Supported benchmarks:
- `hotpotqa`
- `2wikimultihopqa`
- `musique` (default)

### Multi-GPU / distributed

Using Accelerate:

```bash
accelerate launch /path/to/Hi-Q/src/model/hi_q/hi_q.py
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
