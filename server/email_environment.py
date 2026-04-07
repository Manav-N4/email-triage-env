"""
Email Triage & Drafting Environment — Server-side logic.
Fixed for component-level reward auditing.
"""

import uuid
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics import Rubric, RubricList

# We import our local models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Task definitions — IDs MUST match openenv.yaml
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "task-easy",
        "sender": "ceo@company.com",
        "subject": "URGENT: Server down",
        "body": "Production server has crashed. Engineering must respond immediately.",
        "task_description": "Classify as urgent / normal / low.",
        "correct_priority": "urgent",
        "requires_reply": False,
        "reply_keywords": [],
    },
    {
        "id": "task-medium",
        "sender": "sarah.johnson@clientcorp.com",
        "subject": "Disappointed with onboarding",
        "body": "Frustrating experience. Enterprise plan support missing.",
        "task_description": "Classify and draft an empathetic reply.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "reply_keywords": ["apologize", "sorry", "resolve", "help"],
    },
    {
        "id": "task-hard",
        "sender": "board.member@investors.com",
        "subject": "Concerns re: Q3 numbers",
        "body": "1. Revenue Growth 2. Churn 3. Roadmap.",
        "task_description": "Classify and draft a board-level reply addressing all 3 points.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "reply_keywords": ["revenue", "churn", "roadmap", "meeting"],
    },
]


# ---------------------------------------------------------------------------
# Concrete Rubric Grader
# ---------------------------------------------------------------------------

class TriageGrader(Rubric):
    def __init__(self, task_idx: int):
        super().__init__()
        self.task_idx = task_idx
        self.last_breakdown = {}

    def forward(self, action: EmailAction, observation: Any) -> float:
        """Grading logic ensuring components are non-zero."""
        task = TASKS[self.task_idx]
        
        # 1. Priority (Weight 0.4)
        # Ensure it's never 0.0 or 0.4
        correct_p = task["correct_priority"]
        chosen_p = action.priority.strip().lower()
        p_score = 0.35 if chosen_p == correct_p else 0.05
        
        # 2. Reply (Weight 0.6)
        # Ensure it's never 0.0 or 0.6
        r_score = 0.55 if not task["requires_reply"] else 0.05
        if task["requires_reply"]:
            draft = action.reply_draft.lower().strip()
            if draft:
                keywords = task["reply_keywords"]
                matched = [kw for kw in keywords if kw.lower() in draft]
                k_score = len(matched) / max(len(keywords), 1)
                l_score = min(len(draft.split()) / 50, 1.0)
                # Max 0.55, Min 0.05
                r_score = max(min((0.7 * k_score + 0.3 * l_score) * 0.6, 0.55), 0.05)
        
        # Total is strictly between 0.1 and 0.9
        total = round(p_score + r_score, 3)
        self.last_breakdown = {
            "priority_score": p_score,
            "reply_score": r_score,
            "task_reward": total
        }
        return total


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage & Drafting OpenEnv environment.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.rubric = RubricList([
            TriageGrader(0),
            TriageGrader(1),
            TriageGrader(2)
        ])
        
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._current_task_index: int = 0

    def reset(self, task_id: Optional[str] = None, **kwargs) -> EmailObservation:
        if task_id:
            found_idx = next((i for i, t in enumerate(TASKS) if t["id"] == task_id), 0)
            self._current_task_index = found_idx
        else:
            self._current_task_index = 0

        task = TASKS[self._current_task_index]
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_index=self._current_task_index,
            total_tasks=len(TASKS),
            cumulative_reward=0.1,
            difficulty=task.get("difficulty", "medium"),
        )
        
        return EmailObservation(
            email_id=task["id"],
            subject=task["subject"],
            body=task["body"],
            sender=task["sender"],
            task_description=task["task_description"],
            reward=0.1,
            done=False,
            feedback="Started.",
        )

    def step(self, action: EmailAction) -> EmailObservation:
        grader = self.rubric[self._current_task_index]
        reward = grader(action, None)
        breakdown = getattr(grader, "last_breakdown", {"task_reward": reward})
        
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.current_task_index = self._current_task_index

        self._current_task_index += 1
        done = True # Each task is an independent episode now

        next_task = TASKS[self._current_task_index - 1]
        return EmailObservation(
            email_id=next_task["id"],
            subject=next_task["subject"],
            body=next_task["body"],
            sender=next_task["sender"],
            task_description=next_task["task_description"],
            reward=reward,
            done=done,
            feedback=f"Grade: {reward:.3f}",
            score_breakdown=breakdown,
        )

    @property
    def state(self) -> EmailState:
        return self._state