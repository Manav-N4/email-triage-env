#!/usr/bin/env python3
"""
inference.py — Email Triage & Drafting Environment
Strict compliance with sample inference log format.
"""

import os
import sys
import json
import time
from typing import List, Optional
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

from client import EmailTriageEnv
from models import EmailAction

# ---------------------------------------------------------------------------
# Sample Script Logging Helpers (Matches Validator Regex)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# LLM Logic
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, context: str) -> dict:
    prompt = (
        f"You are an email triage assistant. Respond ONLY with JSON.\n"
        f"Fields: 'priority' (urgent/normal/low), 'reply_draft' (string), 'reasoning' (string).\n\n"
        f"{context}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(completion_text := response.choices[0].message.content.strip())


def run_task(llm_client: OpenAI, task_id: str) -> None:
    bench_name = "email-triage-env"
    log_start(task=task_id, env=bench_name, model=MODEL_NAME)
    
    rewards = []
    success = False
    cur_step = 0
    
    try:
        with EmailTriageEnv(base_url=ENV_BASE_URL).sync() as env:
            # 1. Reset
            obs = env.reset(task_id=task_id)
            if hasattr(obs, "observation"):
                obs = obs.observation
            
            done = False
            while not done:
                cur_step += 1
                # 2. Build Context
                context = f"From: {obs.sender}\nSubj: {obs.subject}\n\n{obs.body}\n\nTask: {obs.task_description}"
                
                # 3. Action
                action_dict = call_llm(llm_client, context)
                action = EmailAction(
                    priority=action_dict.get("priority", "normal"),
                    reply_draft=action_dict.get("reply_draft", ""),
                    reasoning=action_dict.get("reasoning", "")
                )
                
                # 4. Step
                step_res = env.step(action)
                reward = step_res.reward
                done = step_res.done
                obs = step_res.observation if hasattr(step_res, "observation") else obs
                
                rewards.append(reward)
                
                # Format action for log (quoted arguments)
                action_str = f'classify("{action.priority}")'
                
                log_step(step=cur_step, action=action_str, reward=reward, done=done, error=None)
                
                # Success usually defined by last reward or cumulative
                success = reward >= 0.1
                
                if cur_step > 10: # Safety break
                    break
            
        # [END] must be emitted AFTER env.close()
        score = rewards[-1] if rewards else 0.01
        log_end(success=success, steps=cur_step, score=score, rewards=rewards)

    except Exception as e:
        # Minimum valid [END] log even on failure
        score = rewards[-1] if rewards else 0.01
        log_end(success=False, steps=cur_step, score=score, rewards=rewards)
        print(f"[DEBUG] Task {task_id} failed: {e}", file=sys.stderr)


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Ensure exactly 3 blocks for the validator
    for task_id in ["email-001", "email-002", "email-003"]:
        run_task(llm_client, task_id)
        time.sleep(1)


if __name__ == "__main__":
    main()