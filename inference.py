#!/usr/bin/env python3
"""
inference.py — Email Triage & Drafting Environment

Runs 3 independent 1-step episodes (one per task).
Each produces exactly one [START] / [STEP] / [END] block.
The validator audits these 3 blocks to confirm "3 tasks with graders".

Required env vars:
  API_BASE_URL  — LLM endpoint  (default: HF router)
  MODEL_NAME    — Model ID      (default: Qwen2.5-72B-Instruct)
  HF_TOKEN      — API key       (required, no default)
  ENV_BASE_URL  — Env server    (default: http://localhost:7860)
"""

import os
import sys
import json
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

from client import EmailTriageEnv
from models import EmailAction

SYSTEM_PROMPT = """\
You are an expert email triage assistant.
Respond ONLY with a valid JSON object — no preamble, no markdown fences.
Required fields:
  "priority"    : one of "urgent", "normal", "low"
  "reply_draft" : professional reply text, or "" if no reply is needed
  "reasoning"   : one sentence explaining your classification
"""

TEMPERATURE = 0.2


def call_llm(client: OpenAI, email_context: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": email_context},
        ],
        max_tokens=800,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def run_task(llm_client: OpenAI, task_id: str) -> None:
    """One independent episode for task_id."""
    with EmailTriageEnv(base_url=ENV_BASE_URL).sync() as env:

        # ── reset ──────────────────────────────────────────────────────────
        reset_result = env.reset(task_id=task_id)
        obs   = reset_result.observation if hasattr(reset_result, "observation") else reset_result
        state = env.state()

        print(f"[START] task_id={obs.email_id} difficulty={state.difficulty}", flush=True)

        # ── build prompt ───────────────────────────────────────────────────
        context = (
            f"From: {obs.sender}\n"
            f"Subject: {obs.subject}\n\n"
            f"{obs.body}\n\n"
            f"Task: {obs.task_description}"
        )

        # ── LLM action ─────────────────────────────────────────────────────
        action_dict = call_llm(llm_client, context)
        action = EmailAction(
            priority=action_dict.get("priority", "normal"),
            reply_draft=action_dict.get("reply_draft", ""),
            reasoning=action_dict.get("reasoning", ""),
        )

        # ── step ───────────────────────────────────────────────────────────
        step_result = env.step(action)
        reward = step_result.reward
        done   = step_result.done

        action_log = json.dumps({
            "priority":   action.priority,
            "reply_words": len(action.reply_draft.split()),
        })
        print(f"[STEP] task_id={obs.email_id} action={action_log} reward={reward:.4f} done={done}", flush=True)
        print(f"[END] task_id={obs.email_id} total_reward={reward:.4f} steps=1", flush=True)


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in ["email-001", "email-002", "email-003"]:
        for attempt in range(3):
            try:
                run_task(llm_client, task_id)
                time.sleep(1)
                break
            except Exception as e:
                print(f"[WARN] {task_id} attempt {attempt+1} failed: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))


if __name__ == "__main__":
    main()