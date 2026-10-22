# Reproducing the coding comparison

The [results file](evaluation/results.json) contains recovered aggregate counts, dataset hashes, per-task pass/fail status and SHA-256 digests of the original result files. These are historical BF16 measurements. The original generated solutions and the complete historical package lockfile are not included.

## Recorded protocol

| Setting | Recorded value |
| --- | --- |
| Parent | `Qwen/Qwen3.8-27B` |
| Parent revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Final format | BF16 |
| Parent runs | 13 October 2026 |
| Final runs | 14 October 2026 |
| Tasks | HumanEval: 164; MBPP: 378 |
| Samples | One solution per task |
| Temperature | 0 |
| Maximum generated tokens | 16,384 |
| Thinking mode | Enabled by default; no `--no-think` flag |
| Generation concurrency | 16 |
| Postprocessing | EvalPlus `sanitize`, using each task's entry point |
| Scoring | EvalPlus, run by a separate unprivileged evaluator user |
| Final truncated generations | HumanEval: 2; MBPP: 0 |

The expected dataset hashes are `fe585eb4df8c88d844eeb463ea4d0302` for HumanEval and `ee43ecabebf20deef4bb776a405ac5b1` for MBPP. Compare them with a new run before interpreting score differences.

The prompt prefix is:

Please provide a self-contained Python script that solves the following problem in a markdown code block:

The generator appends the task prompt inside a fenced code block and sends one user message to an OpenAI-compatible chat completions endpoint. It records sanitized Python solutions for scoring. It retries failed requests up to three times. A request that still fails produces an empty solution and an error in the generation log; inspect the log before accepting a run.

## Current client environment

[requirements-eval.txt](requirements-eval.txt) pins EvalPlus 0.3.1 and HTTPX 0.28.1, the versions recorded in the current environment on 21 October 2026. These pins support a new evaluation client setup. They do not reconstruct all software versions used for the historical measurements.

The serving environment inspected on that date had vLLM 0.31.0, PyTorch 2.13.0 and Transformers 5.17.0. These are current installed versions, rather than a recovered historical lockfile. Model serving needs a separate environment and sufficient hardware.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-eval.txt
```

## Generate and score a new run

The following client commands assume an already running local endpoint at `http://127.0.0.1:8000/v1`, with the final BF16 model registered as `alphaBell-Qwen3.8-27B-Open`. Use the same client settings and backend configuration for the parent comparison.

```bash
mkdir -p outputs
.venv/bin/python evaluation/evalplus_generate.py \
  --base http://127.0.0.1:8000/v1 \
  --model alphaBell-Qwen3.8-27B-Open \
  --dataset humaneval --out outputs/final-humaneval.jsonl \
  --conc 16 --max-tokens 16384
.venv/bin/python evaluation/evalplus_generate.py \
  --base http://127.0.0.1:8000/v1 \
  --model alphaBell-Qwen3.8-27B-Open \
  --dataset mbpp --out outputs/final-mbpp.jsonl \
  --conc 16 --max-tokens 16384
```

The client reads `OPENAI_API_KEY` only when the endpoint requires authentication. It sends that value as a request header and does not embed it in the source.

Score the generated Python in an isolated, unprivileged environment with the evaluation dependencies installed:

```bash
evalplus.evaluate --dataset humaneval --samples outputs/final-humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples outputs/final-mbpp.jsonl
```

Compare both original and extended test-suite results. Report generation errors, truncations, dataset hashes, checkpoint revisions and backend settings with a reproduction. Evaluate FP8 separately before attributing BF16 results to it.

See the [upstream EvalPlus documentation](https://github.com/evalplus/evalplus) for scoring and isolated execution options.
