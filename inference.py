#!/usr/bin/env python3
"""
inference.py — Email Triage & Drafting Environment
Final compliance script — Runs 3 independent episodes for validator success.

Required environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face token (No default)
  ENV_BASE_URL  — Environment server URL
"""

import os
import sys
import json
import time
from openai import OpenAI

# ---------------------------------------------------------------------------
# "I’ve read the sample inference.py and have followed it strictly."
# All LLM calls use the OpenAI client configured via these variables:
# from openai import OpenAI
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")      # No default
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Import the environment client
from client import EmailTriageEnv
from models import EmailAction

TEMPERATURE = 0.2


def call_llm(client: OpenAI, messages: list) -> dict:
    """Uses the OpenAI client to get a structured JSON action."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def run_single_task(llm_client: OpenAI, task_id: str):
    """
    Runs an independent episode for a specific task.
    """
    if LOCAL_IMAGE_NAME:
        env_factory = EmailTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_factory = EmailTriageEnv(base_url=ENV_BASE_URL)

    with env_factory.sync() as env:
        # Reset targeting the specific task ID
        # This creates the [START] / [STEP] / [END] log set the validator needs
        result = env.reset(task_id=task_id)
        obs = result.observation if hasattr(result, "observation") else result
        state = env.state()
        
        # [START] task_id=<id> difficulty=<easy|medium|hard>
        print(f"[START] task_id={obs.email_id} difficulty={state.difficulty}", flush=True)

        # Build context from the task
        prompt = (
            f"From: {obs.sender}\n"
            f"Subject: {obs.subject}\n\n"
            f"{obs.body}\n\n"
            f"--- Task ---\n{obs.task_description}\n\n"
            f"Respond with JSON: {{\"priority\": \"urgent|normal|low\", \"reply_draft\": \"string\"}}"
        )

        messages = [
            {"role": "system", "content": "You are an expert email triage assistant. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        # 1. LLM decide action
        action_dict = call_llm(llm_client, messages)
        action = EmailAction(
            priority=action_dict.get("priority", "normal"),
            reply_draft=action_dict.get("reply_draft", "")
        )

        # 2. Step the Environment
        result = env.step(action)
        reward = result.reward
        done = result.done

        # [STEP] task_id=<id> action=<json> reward=<float> done=<bool>
        action_log = json.dumps({"priority": action.priority, "reply_len": len(action.reply_draft)})
        print(f"[STEP] task_id={obs.email_id} action={action_log} reward={reward:.4f} done={done}", flush=True)

        # [END] task_id=<id> total_reward=<float> steps=<int>
        print(f"[END] task_id={obs.email_id} total_reward={reward:.4f} steps=1", flush=True)


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # LOOP THROUGH ALL 3 TASKS
    # This ensures exactly 3 START / STEP / END logs are audited by the validator.
    tasks_to_run = ["task-easy", "task-medium", "task-hard"]
    
    for tid in tasks_to_run:
        for attempt in range(2):
            try:
                run_single_task(llm_client, tid)
                break
            except Exception as e:
                print(f"Retrying {tid} due to: {e}", file=sys.stderr)
                time.sleep(5)


if __name__ == "__main__":
    main()