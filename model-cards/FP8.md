---
pipeline_tag: image-text-to-text
library_name: transformers
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
datasets:
  - tatsu-lab/alpaca
license: other
license_name: alphabell-research-noncommercial
license_link: ../LICENSE
tags:
  - qwen3_5
  - finetune
  - research
  - refusal-behavior
  - fp8
---

# alphaBell-Qwen3.8-27B-Open-FP8

**A 27B Qwen research derivative for studying refusal behavior and capability tradeoffs, developed by [alphaBell](https://alphabell.com).**

alphaBell-Qwen3.8-27B-Open combines abliteration with supervised fine-tuning to change how the model responds to instructions. This FP8 variant uses blockwise quantization for the main language-model weights. It is intended for noncommercial research into post-training, model behavior and evaluation.

## Model at a glance

| Property | Value |
| --- | --- |
| Lab | [alphaBell](https://alphabell.com) |
| Parent | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Parent revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Post-training | Abliteration followed by supervised fine-tuning |
| Weight format | Safetensors, FP8 E4M3 blockwise quantization with BF16 tensors retained |
| Quantization block size | 128 × 128 |
| Vocabulary, added tokens and chat template | Preserved from the parent checkpoint |
| Intended use | Noncommercial research |

## Evaluation

alphaBell evaluated the final BF16 checkpoint against its Qwen parent with EvalPlus. The table reports one generated solution per task, using greedy decoding with temperature 0 and a maximum of 16,384 generated tokens. The default thinking mode was enabled. Generated solutions were sanitized using EvalPlus before scoring.

| Benchmark | Qwen parent | alphaBell final BF16 | Difference |
| --- | ---: | ---: | ---: |
| HumanEval | 156/164, 95.12% | 157/164, 95.73% | +0.61 percentage points |
| HumanEval+ | 150/164, 91.46% | 148/164, 90.24% | -1.22 percentage points |
| MBPP | 357/378, 94.44% | 347/378, 91.80% | -2.65 percentage points |
| MBPP+ | 306/378, 80.95% | 300/378, 79.37% | -1.59 percentage points |

These results describe a capability tradeoff. The final BF16 checkpoint passes one more HumanEval task and fewer tasks on the other three test suites. This evaluation does not establish a general coding improvement. It is an internal, single-generation comparison: the parent was evaluated on 13 October 2026 and the final BF16 checkpoint on 14 October 2026. Two final HumanEval generations reached the token limit; none did on MBPP.

The evaluation covers text generation. The inherited vision configuration, image and video processors, and configured context length have not been independently evaluated for this derivative. These BF16 results do not establish the performance of this FP8 variant. A complete final-checkpoint FP8 comparison on both suites is not available.

## Training

The model starts from the Qwen parent revision identified above. Post-training combines an abliteration stage with supervised fine-tuning on lab-generated instruction and response pairs. The later fine-tuning stage includes reasoning responses.

Benign replay uses 160 examples from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), sampled with seed 7. The training record does not pin an Alpaca dataset revision. The lab-generated training corpus and a separate held-out behavior set include prompts used to study refusal behavior.

The vocabulary, added tokens and chat template are preserved from Qwen. Tokenizer serialization was updated during export. Image and video processors are preserved. The checkpoint also retains multi-token prediction tensors.

## Intended use and limitations

Use this model to investigate post-training methods, compare instruction-following behavior, and measure how behavioral changes affect other capabilities. Evaluate it against the original Qwen model using the same prompts, inference settings and scoring method.

Changing refusal behavior can also change responses to unsafe or inappropriate requests. More frequent answers do not establish factual accuracy, useful task completion or safety. This model can generate incorrect, biased or harmful content. HumanEval and MBPP results provide a narrow view of its capabilities, and this release has not established suitability for high-stakes or production use.

## License and attribution

This derivative is distributed for noncommercial research under the alphaBell Research Noncommercial terms in `LICENSE`.

The Qwen parent is licensed under Apache 2.0. The Alpaca replay data is licensed under CC BY-NC 4.0, and Stanford's [Alpaca usage notice](https://github.com/tatsu-lab/stanford_alpaca#readme) restricts models trained on that data to research use. The derivative's research and noncommercial terms reflect this training-data provenance. Preserve the upstream Qwen and Alpaca notices when redistributing the corresponding material.

The name “Open” identifies this alphaBell model family. The applicable usage permissions are defined by the release license.

Developed by [alphaBell](https://alphabell.com), building on Qwen and the work of the open model research community.
