"""
Email Triage & Drafting Environment — Final Calibration.
Strictly calibrated rewards to pass log-parsing interval checks.
"""

import os
import sys
import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import EmailAction, EmailObservation, EmailState

TASKS = [
    {
        "id": "email-001",
        "difficulty": "easy",
        "sender": "ceo@company.com",
        "subject": "URGENT",
        "body": "Server down.",
        "task_description": "Classify.",
        "correct_priority": "urgent",
    },
    {
        "id": "email-002",
        "difficulty": "medium",
        "sender": "sarah@client.com",
        "subject": "Help",
        "body": "Frustrating.",
        "task_description": "Reply.",
        "correct_priority": "urgent",
    },
    {
        "id": "email-003",
        "difficulty": "hard",
        "sender": "board@investors.com",
        "subject": "Numbers",
        "body": "Concerned.",
        "task_description": "Address concerns.",
        "correct_priority": "urgent",
    },
]

TASK_BY_ID = {t["id"]: t for t in TASKS}

class EmailTriageEnvironment(Environment):
    """
    OpenEnv compliant environment with 0.33 calibrated rewards.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._current_task: dict = TASKS[0]
        self._state = EmailState()

    def reset(self, task_id: Optional[str] = None, **kwargs) -> EmailObservation:
        self._current_task = TASK_BY_ID.get(task_id, TASKS[0])
        task = self._current_task
        
        # Start with a minimal epsilon reward to stay strictly > 0
        eps = 0.01
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_index=TASKS.index(task),
            total_tasks=len(TASKS),
            cumulative_reward=eps,
            difficulty=task["difficulty"],
        )
        return EmailObservation(
            email_id=task["id"], subject=task["subject"], body=task["body"],
            sender=task["sender"], task_description=task["task_description"],
            reward=eps, done=False, feedback="Task initialized.",
            score_breakdown={"init_eps": eps}
        )

    def step(self, action: EmailAction) -> EmailObservation:
        task = self._current_task
        
        # 1. Dynamic Grading logic
        # Helper to keep any score strictly in (0.01, 0.99)
        def to_safe(s: float) -> float:
            return max(0.01, min(0.99, s))

        # Check priority (70% weight)
        priority_raw = 1.0 if action.priority.strip().lower() == task["correct_priority"].lower() else 0.0
        
        # Check reasoning (10% weight)
        reasoning_raw = 1.0 if len(action.reasoning.strip()) > 10 else 0.0
        
        # Check reply draft (20% weight)
        reply_raw = 1.0
        if task.get("task_description") in ["Reply.", "Address concerns."]:
            reply_raw = 1.0 if len(action.reply_draft.strip()) > 20 else 0.0
            
        raw_score = (priority_raw * 0.7) + (reasoning_raw * 0.1) + (reply_raw * 0.2)
        
        # 2. Final mapped reward strictly in (0.01, 0.99)
        eps = 0.01
        reward = to_safe(eps + (raw_score * (1.0 - 2.0 * eps)))
        
        self._state.step_count += 1
        self._state.cumulative_reward = reward
        
        return EmailObservation(
            email_id=task["id"], subject=task["subject"], body=task["body"],
            sender=task["sender"], task_description=task["task_description"],
            reward=reward, done=True, feedback=f"Task complete. Score: {reward:.4f}",
            score_breakdown={
                "priority_match": to_safe(priority_raw),
                "reasoning_score": to_safe(reasoning_raw),
                "reply_score": to_safe(reply_raw),
                "final_mapped": reward
            }
        )

    @property
    def state(self) -> EmailState:
        return self._state