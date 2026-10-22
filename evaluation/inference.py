#!/usr/bin/env python3
# Copyright 2026 alphaBell. SPDX-License-Identifier: Apache-2.0.
# See ../licenses/CODE_APACHE_2.0.txt for the full license.
"""Run a small, deterministic CPU test of BF16 or a dequantized FP8 checkpoint."""
import argparse
import datetime
import inspect
import json
import os
import time

# Set these before importing PyTorch to keep the test off shared GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_HUB_KERNELS"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer, FineGrainedFP8Config


def use_cpu_reference_kernels():
    """Keep optional CUDA extensions from replacing Transformers' CPU references."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    selected = []
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        function = getattr(modeling_qwen3_5, name, None)
        if function is not None and hasattr(function, "__wrapped__"):
            setattr(modeling_qwen3_5, name, inspect.unwrap(function))
            selected.append(name)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dequantize-fp8", action="store_true",
                        help="Load an FP8 checkpoint as BF16 on CPU; requires accelerate.")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference_functions = use_cpu_reference_kernels()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    loading_options = {}
    if args.dequantize_fp8:
        loading_options["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="sdpa",
        **loading_options,
    ).eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("The smoke test requires all model parameters on CPU.")
    loaded = time.monotonic()
    prompt = "Reply with one short sentence: hello, and what is 2 + 3?"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(json.dumps({
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": "cpu",
        "dtype": "bfloat16",
        "source_precision": "fp8" if args.dequantize_fp8 else "bfloat16",
        "dequantized_on_cpu": args.dequantize_fp8,
        "hub_kernels_enabled": False,
        "unwrapped_reference_functions": reference_functions,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_class": type(model).__name__,
        "trust_remote_code": False,
        "threads": 8,
        "prompt": prompt,
        "thinking": False,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": len(generated),
        "output": text,
        "loading_seconds": round(loaded - started, 3),
        "generation_seconds": round(time.monotonic() - loaded, 3),
        "completed": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
