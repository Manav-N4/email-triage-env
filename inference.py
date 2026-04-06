#!/usr/bin/env python3
"""
inference.py — Email Triage & Drafting Environment
Baseline inference script using an LLM agent via OpenAI-compatible API.

Required environment variables:
  API_BASE_URL  — LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier   (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — Hugging Face token (used as API key)

Structured stdout format (DO NOT change):
  [START] task_id=<id> difficulty=<easy|medium|hard>
  [STEP]  task_id=<id> action=<json> reward=<float> done=<bool>
  [END]   task_id=<id> total_reward=<float> steps=<int>

Run:
  python inference.py
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_EPISODE = 10   # safety cap
TEMPERATURE           = 0.2

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert email triage assistant working inside an RL training environment.

For each email you receive, you must respond ONLY with a valid JSON object with these exact fields:
{
  "priority": "<urgent|normal|low>",
  "reply_draft": "<your professional reply or empty string if no reply needed>",
  "reasoning": "<brief explanation of your classification>"
}

Priority rules:
- urgent: Requires same-day response. C-suite, clients, incidents, deadlines.
- normal: Standard business communication. Respond within 1–2 days.
- low: FYI, newsletters, no action required.

Reply rules:
- If task_description says no reply is needed, set reply_draft to ""
- Otherwise write a professional reply that directly addresses the concerns
- Use names when available, match the formality of the sender
- Be specific — vague replies score lower than concrete ones

Respond ONLY with the JSON object. No preamble, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# HTTP helpers for talking to the env server directly (REST fallback)
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_state() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> dict:
    """
    Extract JSON from LLM output. Falls back to a safe default on parse error.
    """
    try:
        # Strip markdown fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1])
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        # Safe fallback
        return {"priority": "normal", "reply_draft": "", "reasoning": "parse error"}


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI) -> dict:
    """
    Run one full episode (3 tasks). Returns summary dict.
    """
    # --- reset ---
    obs = env_reset()
    state = env_state()

    task_id      = obs.get("email_id", "unknown")
    difficulty   = obs.get("difficulty", "easy") or state.get("difficulty", "easy")
    total_reward = 0.0
    step_count   = 0

    print(f"[START] task_id={task_id} difficulty={difficulty}", flush=True)

    while not obs.get("done", False) and step_count < MAX_STEPS_PER_EPISODE:
        # Build prompt from current observation
        email_context = (
            f"From: {obs.get('sender', '')}\n"
            f"Subject: {obs.get('subject', '')}\n\n"
            f"{obs.get('body', '')}\n\n"
            f"--- Task ---\n{obs.get('task_description', '')}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": email_context},
        ]

        # LLM decides the action
        raw_response = call_llm(client, messages)
        action       = parse_action(raw_response)

        # Validate / sanitise
        action["priority"]    = action.get("priority", "normal").lower().strip()
        action["reply_draft"] = action.get("reply_draft", "")
        action["reasoning"]   = action.get("reasoning", "")

        # Step the environment
        result = env_step(action)
        reward = result.get("reward", 0.0)
        done   = result.get("done",   False)

        # --- REQUIRED structured log ---
        action_log = json.dumps({
            "priority":    action["priority"],
            "reply_words": len(action["reply_draft"].split()),
        })
        print(
            f"[STEP]  task_id={task_id} "
            f"action={action_log} "
            f"reward={reward:.4f} "
            f"done={done}",
            flush=True,
        )

        total_reward += reward
        step_count   += 1

        # Move to next observation
        obs = result.get("observation", result)
        task_id    = obs.get("email_id", task_id)
        difficulty = obs.get("difficulty", difficulty) or difficulty

        # Delay to avoid rate limits
        time.sleep(0.5)

    print(
        f"[END]   task_id={task_id} "
        f"total_reward={total_reward:.4f} "
        f"steps={step_count}",
        flush=True,
    )

    return {
        "total_reward": total_reward,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"Email Triage Inference")
    print(f"Model    : {MODEL_NAME}")
    print(f"Endpoint : {API_BASE_URL}")
    print(f"Env URL  : {ENV_BASE_URL}")
    print("=" * 60, flush=True)

    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    summary = run_episode(llm_client)
    print("\n--- Summary ---")
    print(f"Total reward : {summary['total_reward']:.4f}")
    print(f"Steps taken  : {summary['steps']}")
    print(f"Score        : {summary['total_reward'] / max(summary['steps'], 1):.4f} per step")


if __name__ == "__main__":
    main()