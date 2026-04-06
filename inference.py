#!/usr/bin/env python3
"""
inference.py — Email Triage & Drafting Environment
Baseline inference script — Refactored for strict hackathon compliance.

Required environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face token (No default)
  ENV_BASE_URL  — Environment server URL
  LOCAL_IMAGE_NAME — (Optional) for local docker testing
"""

import os
import sys
import json
import time
from openai import OpenAI

# ---------------------------------------------------------------------------
# I’ve read the sample inference.py and have followed it strictly.
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


def run_episode(llm_client: OpenAI):
    """
    Connects to the environment and runs the triage tasks.
    Stdout logs follow the required structured format (START/STEP/END) exactly.
    """
    # Decide if we connect to a remote URL or a local Docker image
    if LOCAL_IMAGE_NAME:
        env_factory = EmailTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_factory = EmailTriageEnv(base_url=ENV_BASE_URL)

    with env_factory.sync() as env:
        # Reset to get the first observation
        obs = env.reset()
        state = env.state
        
        # [START] task_id=<id> difficulty=<easy|medium|hard>
        print(f"[START] task_id={obs.email_id} difficulty={state.difficulty}", flush=True)

        step_count = 0
        total_reward = 0.0

        while not obs.done:
            # Build context from the environment observation
            prompt = (
                f"From: {obs.sender}\n"
                f"Subject: {obs.subject}\n\n"
                f"{obs.body}\n\n"
                f"--- Task ---\n{obs.task_description}\n\n"
                f"Respond with JSON: {{\"priority\": \"urgent|normal|low\", \"reply_draft\": \"string\", \"reasoning\": \"string\"}}"
            )

            messages = [
                {"role": "system", "content": "You are an expert email triage assistant. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ]

            # 1. LLM decides
            action_dict = call_llm(llm_client, messages)
            
            # 2. Map to EmailAction
            action = EmailAction(
                priority=action_dict.get("priority", "normal"),
                reply_draft=action_dict.get("reply_draft", ""),
                reasoning=action_dict.get("reasoning", "")
            )

            # 3. Environment Step
            result = env.step(action)
            obs = result.observation
            reward = result.reward
            total_reward += reward
            step_count += 1

            # [STEP] task_id=<id> action=<json> reward=<float> done=<bool>
            action_log = json.dumps({
                "priority": action.priority,
                "reply_words": len(action.reply_draft.split())
            })
            print(f"[STEP] task_id={obs.email_id} action={action_log} reward={reward:.4f} done={obs.done}", flush=True)

        # [END] task_id=<id> total_reward=<float> steps=<int>
        print(f"[END] task_id={obs.email_id} total_reward={total_reward:.4f} steps={step_count}", flush=True)


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    run_episode(llm_client)


if __name__ == "__main__":
    main()