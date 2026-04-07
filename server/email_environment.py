"""
Email Triage & Drafting Environment — Server-side logic.
Fixed with Goldilocks Headroom scoring (0.5 to 0.9 window).
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
        "subject": "URGENT: Server down — all hands needed NOW",
        "body": "Our primary production server has crashed. Engineering must respond immediately.",
        "task_description": "Classify the priority of this email (urgent / normal / low).",
        "correct_priority": "urgent",
        "requires_reply": False,
        "keywords": [],
    },
    {
        "id": "email-002",
        "difficulty": "medium",
        "sender": "sarah.johnson@clientcorp.com",
        "subject": "Disappointed with onboarding experience",
        "body": "Frustrating experience. Two weeks and doc is outdated. Escalating.",
        "task_description": "Classify and draft a professional reply apologizing and offering help.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["apologize", "sorry", "resolve", "help"],
    },
    {
        "id": "email-003",
        "difficulty": "hard",
        "sender": "board.member@investors.com",
        "subject": "Concerns re: Q3 numbers and strategic direction",
        "body": "1. Revenue growth slowed. 2. Churn increased. 3. AI roadmap.",
        "task_description": "Address each of the 3 concerns individually in a board-level reply.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["revenue", "churn", "roadmap", "call"],
    },
]

TASK_BY_ID = {t["id"]: t for t in TASKS}

def _grade(action: EmailAction, task: dict) -> float:
    """
    Returns a reward between 0.1 and 0.4.
    Neutral Start (0.5) + Step (0.1..0.4) = Episode Total (0.6..0.9).
    """
    # 1. Priority (Max 0.15, Min 0.05)
    chosen = action.priority.strip().lower()
    p_score = 0.15 if chosen == task["correct_priority"] else 0.05
    
    # 2. Reply (Max 0.25, Min 0.05)
    r_score = 0.25 if not task["requires_reply"] else 0.05
    if task["requires_reply"]:
        draft = (action.reply_draft or "").strip()
        if draft:
            keywords = task["keywords"]
            matched = sum(1 for kw in keywords if kw.lower() in draft.lower())
            ratio = matched / max(len(keywords), 1)
            # Map [0, 1] to [0.05, 0.25]
            r_score = 0.05 + (ratio * 0.20)
    
    return round(p_score + r_score, 3)

class EmailTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._current_task: dict = TASKS[0]
        self._state = EmailState()

    def reset(self, task_id: Optional[str] = None, **kwargs) -> EmailObservation:
        self._current_task = TASK_BY_ID.get(task_id, TASKS[0])
        task = self._current_task
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_index=TASKS.index(task),
            total_tasks=len(TASKS),
            cumulative_reward=0.5, # Mid-interval start
            difficulty=task["difficulty"],
        )
        return EmailObservation(
            email_id=task["id"], subject=task["subject"], body=task["body"],
            sender=task["sender"], task_description=task["task_description"],
            reward=0.5, done=False, feedback="Episode started."
        )

    def step(self, action: EmailAction) -> EmailObservation:
        task = self._current_task
        reward = _grade(action, task)
        
        self._state.step_count += 1
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 3)
        
        feedback = f"task={task['id']} step_reward={reward} total={self._state.cumulative_reward}"
        
        return EmailObservation(
            email_id=task["id"], subject=task["subject"], body=task["body"],
            sender=task["sender"], task_description=task["task_description"],
            reward=reward, done=True, feedback=feedback
        )

    @property
    def state(self) -> EmailState:
        return self._state