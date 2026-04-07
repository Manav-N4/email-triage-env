"""
Email Triage & Drafting Environment — Final Fusion.
Claude's Task Content + Robust Rubric System.
"""

import uuid
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics import Rubric, RubricDict

# We import our local models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Task definitions (Claude's Content)
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "email-001",
        "sender": "ceo@company.com",
        "subject": "URGENT: Server down — all hands needed NOW",
        "body": "Our primary production server has crashed. Engineering must respond immediately.",
        "task_description": "Classify priority. No reply needed.",
        "correct_priority": "urgent",
        "requires_reply": False,
        "keywords": [],
    },
    {
        "id": "email-002",
        "sender": "sarah.johnson@clientcorp.com",
        "subject": "Disappointed with onboarding experience",
        "body": "Frustrating experience. Two weeks and doc is outdated. Escalating.",
        "task_description": "Classify and draft an empathetic reply.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["apologize", "sorry", "resolve", "help", "contact"],
    },
    {
        "id": "email-003",
        "sender": "board.member@investors.com",
        "subject": "Concerns re: Q3 numbers and strategic direction",
        "body": "1. Revenue growth slowed. 2. Churn increased. 3. AI roadmap.",
        "task_description": "Classify and draft a board-level reply addressing all 3 concerns.",
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["revenue", "churn", "roadmap", "call", "thursday"],
    },
]


# ---------------------------------------------------------------------------
# Robust Rubric System
# ---------------------------------------------------------------------------

class TriageGrader(Rubric):
    def __init__(self, task_idx: int):
        super().__init__()
        self.task_idx = task_idx

    def forward(self, action: EmailAction, observation: Any) -> float:
        """
        Universal Offset logic:
        - Tiny Base (0.1) satisfies 'strictly between 0 and 1' for every grader.
        - Sum stays <= 0.9.
        """
        score = 0.1 # Strictly between 0 and 1
        
        current_id = getattr(observation, "email_id", "")
        if current_id == TASKS[self.task_idx]["id"]:
            # Active task gets a bonus
            task = TASKS[self.task_idx]
            p_correct = (action.priority.strip().lower() == task["correct_priority"])
            p_bonus = 0.2 if p_correct else 0.05
            
            r_bonus = 0.05
            if task["requires_reply"]:
                draft = action.reply_draft.lower()
                matched = [k for k in task["keywords"] if k.lower() in draft]
                k_ratio = len(matched) / max(len(task["keywords"]), 1)
                r_bonus = 0.05 + (k_ratio * 0.45)
            else:
                r_bonus = 0.5
            
            # Max contribution = 0.2 + 0.5 = 0.7
            score = round(p_bonus + r_bonus, 3) # Max 0.7
            
        return score


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage & Drafting OpenEnv environment.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # 3 robust graders for Phase 2 audit
        self.rubric = RubricDict({
            "grader-001": TriageGrader(0),
            "grader-002": TriageGrader(1),
            "grader-003": TriageGrader(2),
        })
        self._current_task_index = 0
        self._state = EmailState()

    def reset(self, task_id: Optional[str] = None, **kwargs) -> EmailObservation:
        self._current_task_index = next((i for i, t in enumerate(TASKS) if t["id"] == task_id), 0)
        task = TASKS[self._current_task_index]
        self._state = EmailState(
            current_task_index=self._current_task_index,
            difficulty="medium",
            cumulative_reward=0.01
        )
        return EmailObservation(
            email_id=task["id"], subject=task["subject"], body=task["body"],
            sender=task["sender"], task_description=task["task_description"],
            reward=0.01, done=False, feedback="Started."
        )

    def step(self, action: EmailAction) -> EmailObservation:
        task = TASKS[self._current_task_index]
        obs_for_rubric = EmailObservation(
            email_id=task["id"], subject="", body="", sender="", task_description="", reward=0.01
        )
        # Sum of 3 graders: (0.1) + (0.1) + (0.1 to 0.7) = 0.3 to 0.9
        reward = self.rubric(action, obs_for_rubric)
        
        self._state.cumulative_reward += reward
        # Mark as done to support 3-episode validation
        return EmailObservation(
            email_id=task["id"], subject="", body="", sender="", task_description="",
            reward=reward, done=True, feedback=f"Final: {reward:.3f}"
        )

    @property
    def state(self) -> EmailState:
        return self._state