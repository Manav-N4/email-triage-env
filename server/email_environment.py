"""
Email Triage & Drafting Environment — Server-side logic.
Fixed with concrete Rubric implementation.
"""

import uuid
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics import Rubric, RubricDict

# We import our local models using relative path
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
        "task_description": "Classify as urgent / normal / low. No reply needed.",
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
# Simple Grader implementation (Concrete Rubric)
# ---------------------------------------------------------------------------

class SimpleGrader(Rubric):
    def __init__(self, task_idx: int):
        super().__init__()
        self.task_idx = task_idx

    def forward(self, action: EmailAction, observation: Any) -> float:
        """Grading logic called by the framework."""
        task = TASKS[self.task_idx]
        
        # 1. Priority Score (0.4)
        correct_p = task["correct_priority"]
        chosen_p = action.priority.strip().lower()
        p_score = 0.4 if chosen_p == correct_p else 0.0
        
        # 2. Reply Score (0.6)
        r_score = 0.0
        if not task["requires_reply"]:
            r_score = 0.6
        else:
            draft = action.reply_draft.lower()
            if draft.strip():
                keywords = task["reply_keywords"]
                matched = [kw for kw in keywords if kw.lower() in draft]
                keyword_score = len(matched) / max(len(keywords), 1)
                
                word_count = len(draft.split())
                length_score = min(word_count / 50, 1.0)
                
                r_score = (0.7 * keyword_score + 0.3 * length_score) * 0.6
        
        # HACKATHON REQUIREMENT: Reward must be strictly > 0 and < 1.
        # Clip to (0.01, 0.99)
        total = round(min(max(p_score + r_score, 0.01), 0.99), 3)
        return total


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage & Drafting OpenEnv environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # Must be initialized correctly with a concrete Rubric container
        self.rubric = RubricDict({
            "task-easy":   SimpleGrader(0),
            "task-medium": SimpleGrader(1),
            "task-hard":   SimpleGrader(2),
        })
        
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._current_task_index: int = 0

    def reset(self) -> EmailObservation:
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_index=0,
            total_tasks=len(TASKS),
            cumulative_reward=0.5,
            difficulty="easy",
        )
        self._current_task_index = 0
        task = TASKS[0]
        return EmailObservation(
            email_id=task["id"],
            subject=task["subject"],
            body=task["body"],
            sender=task["sender"],
            task_description=task["task_description"],
            reward=0.5,
            done=False,
            feedback="Episode started.",
        )

    def step(self, action: EmailAction) -> EmailObservation:
        # Use our concrete grader for the reward
        grader = self.rubric[TASKS[self._current_task_index]["id"]]
        reward = grader.forward(action, None)
        
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.current_task_index = self._current_task_index

        # Advance to next task
        self._current_task_index += 1
        done = self._current_task_index >= len(TASKS)

        if done:
            next_task = TASKS[len(TASKS)-1]
            feedback = f"Episode complete! Reward: {reward:.3f}"
        else:
            next_task = TASKS[self._current_task_index]
            feedback = f"Task graded. Reward: {reward:.3f}"
            self._state.difficulty = next_task.get("difficulty", "medium")

        return EmailObservation(
            email_id=next_task["id"] if not done else "done",
            subject=next_task["subject"] if not done else "",
            body=next_task["body"] if not done else "",
            sender=next_task["sender"] if not done else "",
            task_description=next_task["task_description"] if not done else "",
            reward=reward,
            done=done,
            feedback=feedback,
        )

    @property
    def state(self) -> EmailState:
        return self._state