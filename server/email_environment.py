"""
Email Triage & Drafting Environment.

Reward contract (strictly enforced):
  - Every reward value emitted — from reset() or step() — is in the open
    interval (0.0, 1.0). The _safe() helper is the single chokepoint that
    guarantees this. No value ever touches 0.0 or 1.0.

Grading (no external Rubric dependency — pure Python):
  - priority_score : 0.35 if correct, 0.05 if wrong  -> never 0 or 1
  - reply_score    : 0.05 ... 0.55 based on keyword coverage + length
  - total          : priority_score + reply_score -> range [0.10, 0.90]
  - _safe() clamps to [0.01, 0.99] as a final safety net
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


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "email-001",
        "difficulty": "easy",
        "sender": "ceo@company.com",
        "subject": "URGENT: Server down — all hands needed NOW",
        "body": (
            "Our primary production server has crashed. The entire platform is "
            "offline. Customers cannot access their accounts. Engineering must "
            "respond immediately. This is a P0 incident. Get on a call NOW."
        ),
        "task_description": (
            "Classify the priority of this email (urgent / normal / low). "
            "No reply is needed for internal incident alerts."
        ),
        "correct_priority": "urgent",
        "requires_reply": False,
        "keywords": [],
    },
    {
        "id": "email-002",
        "difficulty": "medium",
        "sender": "sarah.johnson@clientcorp.com",
        "subject": "Disappointed with onboarding experience",
        "body": (
            "Hi,\n\nI've been trying to get my team set up on your platform for "
            "two weeks and the experience has been really frustrating. Our "
            "dedicated onboarding rep hasn't responded to emails in 5 days, and "
            "the documentation is outdated. We're paying for the Enterprise plan "
            "and expect better support.\n\nPlease escalate this.\n\n"
            "Sarah Johnson\nHead of Operations, ClientCorp"
        ),
        "task_description": (
            "Classify the priority AND draft a professional reply. "
            "Acknowledge the frustration, apologise, commit to specific follow-up, "
            "and provide a direct contact."
        ),
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["apologize", "sorry", "follow up", "contact", "resolve", "help"],
    },
    {
        "id": "email-003",
        "difficulty": "hard",
        "sender": "board.member@investors.com",
        "subject": "Concerns re: Q3 numbers and strategic direction",
        "body": (
            "Hello,\n\nI've reviewed the Q3 report and want to share concerns:\n\n"
            "1. Revenue growth slowed to 8% vs 15% projected.\n"
            "2. Customer churn rose from 3.2% to 4.8%.\n"
            "3. The product roadmap lacks AI-native features competitors are shipping.\n\n"
            "I'd welcome a 30-minute call before Thursday.\n\n"
            "Regards,\nDavid Mercer\nBoard Member"
        ),
        "task_description": (
            "Classify priority AND draft a board-level reply. Address each of the "
            "3 concerns individually. Maintain a confident but not defensive tone. "
            "Propose a specific call time before Thursday."
        ),
        "correct_priority": "urgent",
        "requires_reply": True,
        "keywords": ["revenue", "churn", "roadmap", "call", "thursday", "strategy"],
    },
]

TASK_BY_ID = {t["id"]: t for t in TASKS}


# ---------------------------------------------------------------------------
# Reward helpers — single chokepoint guarantees (0, 1) on every value
# ---------------------------------------------------------------------------

def _safe(value: float) -> float:
    """Clamp to [0.01, 0.90] — ensure sum never hits 1.0."""
    return round(max(0.01, min(0.90, value)), 4)


def _grade(action: EmailAction, task: dict) -> float:
    """
    Compute reward for one task. Min=0.10, max=0.90 before _safe().

    Priority component  (0.05 or 0.35):
      correct -> 0.35
      wrong   -> 0.05   (non-zero so total never hits 0)

    Reply component  (0.05 to 0.55):
      no reply needed -> flat 0.45
      empty draft     -> 0.05
      has draft       -> 0.05 + kw_ratio*0.40 + length_bonus(max 0.10)

    Combined min: 0.05 + 0.05 = 0.10
    Combined max: 0.35 + 0.55 = 0.90
    """
    # --- Priority ---
    chosen = action.priority.strip().lower()
    priority_score = 0.35 if chosen == task["correct_priority"] else 0.05

    # --- Reply ---
    if not task["requires_reply"]:
        reply_score = 0.45
    else:
        draft = (action.reply_draft or "").strip()
        if not draft:
            reply_score = 0.05
        else:
            keywords = task["keywords"]
            if keywords:
                matched = sum(1 for kw in keywords if kw.lower() in draft.lower())
                kw_ratio = matched / len(keywords)
            else:
                kw_ratio = 1.0

            words = len(draft.split())
            length_bonus = min(words / 100.0, 1.0) * 0.10
            reply_score = 0.05 + kw_ratio * 0.40 + length_bonus

    return _safe(priority_score + reply_score)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage & Drafting OpenEnv environment.

    Each episode targets one task (selected via reset(task_id=...)).
    The episode ends after a single step() — done=True.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._current_task: dict = TASKS[0]
        self._state = EmailState()

    def reset(self, task_id: Optional[str] = None, **kwargs) -> EmailObservation:
        if task_id and task_id in TASK_BY_ID:
            self._current_task = TASK_BY_ID[task_id]
        else:
            self._current_task = TASKS[0]

        task = self._current_task
        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_index=TASKS.index(task),
            total_tasks=len(TASKS),
            cumulative_reward=0.01,
            difficulty=task["difficulty"],
        )

        return EmailObservation(
            email_id=task["id"],
            subject=task["subject"],
            body=task["body"],
            sender=task["sender"],
            task_description=task["task_description"],
            reward=0.01,
            done=False,
            feedback="Episode started. Read the email and respond.",
        )

    def step(self, action: EmailAction) -> EmailObservation:
        task = self._current_task
        reward = _grade(action, task)   # guaranteed in [0.01, 0.99]

        self._state.step_count += 1
        self._state.cumulative_reward = _safe(
            self._state.cumulative_reward + reward
        )

        correct_p = action.priority.strip().lower() == task["correct_priority"]
        feedback = (
            f"priority={'correct' if correct_p else 'wrong'} "
            f"reply_words={len((action.reply_draft or '').split())} "
            f"reward={reward}"
        )

        return EmailObservation(
            email_id=task["id"],
            subject=task["subject"],
            body=task["body"],
            sender=task["sender"],
            task_description=task["task_description"],
            reward=reward,
            done=True,
            feedback=feedback,
        )

    @property
    def state(self) -> EmailState:
        return self._state