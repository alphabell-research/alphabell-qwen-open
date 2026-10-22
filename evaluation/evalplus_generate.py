#!/usr/bin/env python3
# Copyright 2026 alphaBell
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See ../licenses/CODE_APACHE_2.0.txt for the full license.
"""Parallel EvalPlus generation against an OpenAI-compatible server (greedy, EvalPlus prompt format).

Writes sanitized samples for HumanEval+ or MBPP+; score them with
  evalplus.evaluate --dataset humaneval --samples OUT.jsonl
Usage: evalpar.py --base URL --model NAME --dataset humaneval|mbpp --out OUT.jsonl [--conc 32] [--max-tokens 16384]
"""
import argparse, asyncio, json, os, time
import httpx
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.sanitize import sanitize

PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


async def solve(client, args, task_id, task, sem, stats):
    msg = f"{PREFIX}\n```\n{task['prompt'].strip()}\n```\n"
    payload = {"model": args.model, "messages": [{"role": "user", "content": msg}],
               "temperature": 0.0, "max_tokens": args.max_tokens}
    if args.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(f"{args.base}/chat/completions", json=payload, timeout=3600)
                r.raise_for_status()
                d = r.json()
                break
            except Exception as e:  # retry transient server errors
                if attempt == 2:
                    print(f"{task_id}: {e}")
                    d = {"choices": [{"message": {"content": ""}, "finish_reason": "error"}], "usage": {}}
                await asyncio.sleep(5)
    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    stats["tokens"] += (d.get("usage") or {}).get("completion_tokens", 0)
    stats["truncated"] += ch.get("finish_reason") == "length"
    return {"task_id": task_id, "solution": sanitize(text, entrypoint=task["entry_point"])}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conc", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--no-think", action="store_true")
    args = p.parse_args()
    problems = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    sem = asyncio.Semaphore(args.conc)
    stats = {"tokens": 0, "truncated": 0}
    t0 = time.time()
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"} if os.environ.get("OPENAI_API_KEY") else {}
    async with httpx.AsyncClient(headers=headers) as client:
        rows = await asyncio.gather(*(solve(client, args, k, v, sem, stats) for k, v in problems.items()))
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    dt = time.time() - t0
    print(json.dumps({"model": args.model, "dataset": args.dataset, "problems": len(rows), "seconds": round(dt),
                      "completion_tokens": stats["tokens"], "truncated": stats["truncated"],
                      "aggregate_tps": round(stats["tokens"] / dt, 1)}))


if __name__ == "__main__":
    asyncio.run(main())
