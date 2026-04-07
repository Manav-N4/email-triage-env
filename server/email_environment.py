"""
Email Triage & Drafting Environment — Server-side logic.

Three tasks of increasing difficulty:
  Task 0 (easy)   — Classify a clearly urgent email, no reply needed
  Task 1 (medium) — Classify + draft a professional reply to a client complaint
  Task 2 (hard)   — Classify an ambiguous escalation + draft a nuanced reply
                    that addresses multiple concerns with appropriate tone
"""

import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# We import our local models using relative path so the server can find them
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import EmailAction, EmailObservation, EmailState


# ---------------------------------------------------------------------------
# Task definitions — IDs MUST match openenv.yaml
# ---------------------------------------------------------------------------

TASKS = [
    {
        "difficulty": "easy",
        "email_id": "task-easy",
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
        "reply_keywords": [],
        "reply_tone": None,
    },
    {
        "difficulty": "medium",
        "email_id": "task-medium",
        "sender": "sarah.johnson@clientcorp.com",
        "subject": "Disappointed with onboarding experience",
        "body": (
            "Hi,\n\n"
            "I've been trying to get my team set up on your platform for the past "
            "two weeks and the experience has been really frustrating. Our "
            "dedicated onboarding rep hasn't responded to emails in 5 days, and "
            "the documentation is outdated. We're paying for the Enterprise plan "
            "and expect better support.\n\n"
            "Please escalate this to someone who can actually help us.\n\n"
            "Sarah Johnson\nHead of Operations, ClientCorp"
        ),
        "task_description": (
            "Classify the priority of this email AND draft a professional reply. "
            "The reply should: acknowledge the frustration, apologize sincerely, "
            "commit to a specific follow-up action, and provide a direct contact."
        ),
        "correct_priority": "urgent",
        "requires_reply": True,
        "reply_keywords": ["apologize", "sorry", "follow up", "contact", "resolve", "help"],
        "reply_tone": "professional and empathetic",
    },
    {
        "difficulty": "hard",
        "email_id": "task-hard",
        "sender": "board.member@investors.com",
        "subject": "Concerns re: Q3 numbers and strategic direction",
        "body": (
            "Hello,\n\n"
            "I've reviewed the Q3 financial report and want to share a few concerns "
            "before our board meeting next Thursday.\n\n"
            "1. Revenue growth has slowed to 8% vs the 15% projected.\n"
            "2. Customer churn increased from 3.2% to 4.8% — this is alarming.\n"
            "3. The new product roadmap presented last month seems to lack the "
            "AI-native features that competitors are shipping aggressively.\n\n"
            "I'm not suggesting panic, but I do think we need a candid conversation "
            "about whether the current strategy is the right one. I'd welcome a "
            "30-minute call before Thursday if possible.\n\n"
            "Regards,\nDavid Mercer\nBoard Member"
        ),
        "task_description": (
            "Classify the priority of this email AND draft a board-level reply. "
            "The reply must: acknowledge each of the 3 concerns individually, "
            "maintain a confident but not defensive tone, propose a specific call "
            "time, and reassure without making unverifiable promises."
        ),
        "correct_priority": "urgent",
        "requires_reply": True,
        "reply_keywords": [
            "revenue", "churn", "roadmap",  # must address each concern
            "call", "thursday", "meeting",  # must propose a meeting
            "strategy", "confident",        # must address strategic concern
        ],
        "reply_tone": "executive and measured",
    },
]


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_priority(action: EmailAction, task: dict) -> tuple[float, str]:
    """Return (score 0.0-0.4, feedback)."""
    correct = task["correct_priority"]
    chosen = action.priority.strip().lower()
    if chosen == correct:
        return 0.4, f"✅ Priority correctly classified as '{correct}'."
    else:
        return 0.0, f"❌ Priority should be '{correct}', got '{chosen}'."


def grade_reply(action: EmailAction, task: dict) -> tuple[float, str]:
    """Return (score 0.0-0.6, feedback). 0.6 total for reply quality."""
    if not task["requires_reply"]:
        return 0.6, "ℹ️ No reply required for this task — full reply score awarded."

    draft = action.reply_draft.lower()
    if not draft.strip():
        return 0.0, "❌ Reply draft is empty."

    keywords = task["reply_keywords"]
    matched = [kw for kw in keywords if kw.lower() in draft]
    keyword_score = len(matched) / max(len(keywords), 1)

    # Length check: a meaningful reply is at least 50 words
    word_count = len(draft.split())
    length_score = min(word_count / 80, 1.0)  # full marks at 80+ words

    # Combine: 70% keyword coverage, 30% length
    raw = 0.7 * keyword_score + 0.3 * length_score
    final_score = round(raw * 0.6, 3)  # scale to 0.0–0.6

    feedback_parts = [
        f"Reply keywords matched: {matched} ({len(matched)}/{len(keywords)})",
        f"Word count: {word_count} ({'✅ good' if word_count >= 50 else '⚠️ too short'})",
    ]
    if final_score >= 0.5:
        feedback_parts.append("✅ Strong reply draft.")
    elif final_score >= 0.3:
        feedback_parts.append("⚠️ Acceptable draft but missing some key elements.")
    else:
        feedback_parts.append("❌ Reply draft needs more detail and coverage of key points.")

    return final_score, " | ".join(feedback_parts)


def compute_reward(action: EmailAction, task: dict) -> tuple[float, dict, str]:
    """
    Returns (total_reward strictly within (0, 1), score_breakdown, feedback_string).
    """
    p_score, p_fb = grade_priority(action, task)
    r_score, r_fb = grade_reply(action, task)
    
    # Raw total is in [0.0, 1.0]
    raw_total = p_score + r_score
    
    # HACKATHON REQUIREMENT: Reward must be strictly > 0 and < 1.
    # We clip to [0.01, 0.99] to ensure we never hit 0.0 or 1.0.
    total = round(min(max(raw_total, 0.01), 0.99), 3)
    
    breakdown = {
        "priority_score": p_score,
        "reply_score": r_score,
        "total": total,
    }
    feedback = f"{p_fb}\n{r_fb}\n→ Total reward: {total:.3f} / 1.000"
    return total, breakdown, feedback


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(Environment):
    """
    Email Triage & Drafting OpenEnv environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
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
            cumulative_reward=0.01, # Start with tiny reward
            difficulty=TASKS[0]["difficulty"],
        )
        self._current_task_index = 0
        task = TASKS[0]
        return EmailObservation(
            email_id=task["email_id"],
            subject=task["subject"],
            body=task["body"],
            sender=task["sender"],
            task_description=task["task_description"],
            reward=0.01, # Validator hack: strictly between 0 and 1
            done=False,
            feedback="Episode started. Classify and/or reply to the email.",
            score_breakdown={},
        )

    def step(self, action: EmailAction) -> EmailObservation:
        task = TASKS[self._current_task_index]
        reward, breakdown, feedback = compute_reward(action, task)

        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.current_task_index = self._current_task_index

        # Advance to next task
        self._current_task_index += 1
        done = self._current_task_index >= len(TASKS)

        if done:
            next_task = task  # show same task data, but done=True
            self._state.difficulty = task["difficulty"]
            feedback += f"\n\n🏁 Episode complete! Cumulative reward: {self._state.cumulative_reward:.3f}"
        else:
            next_task = TASKS[self._current_task_index]
            self._state.difficulty = next_task["difficulty"]

        return EmailObservation(
            email_id=next_task["email_id"] if not done else "done",
            subject=next_task["subject"] if not done else "",
            body=next_task["body"] if not done else "",
            sender=next_task["sender"] if not done else "",
            task_description=next_task["task_description"] if not done else "",
            reward=reward,
            done=done,
            feedback=feedback,
            score_breakdown=breakdown,
        )

    @property
    def state(self) -> EmailState:
        return self._state