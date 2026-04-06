"""
Email Triage & Drafting Environment — Type-safe models.

Action  : What the agent sends (priority label + reply draft)
Observation : What the agent receives (email content + feedback)
State   : Episode-level metadata
"""

from dataclasses import dataclass, field
from typing import Optional
from openenv.core.env_server.types import State  # re-exported as-is


@dataclass
class EmailAction:
    """
    The agent's response to a presented email.

    Fields
    ------
    priority : str
        One of "urgent", "normal", "low"
    reply_draft : str
        The agent's suggested reply (may be empty string for low-priority)
    reasoning : str
        Optional chain-of-thought the agent provides (not scored, helps debug)
    """
    priority: str
    reply_draft: str
    reasoning: str = ""


@dataclass
class EmailObservation:
    """
    What the environment sends back after reset() or step().

    Fields
    ------
    email_id        : Unique ID for the current email task
    subject         : Email subject line
    body            : Email body text
    sender          : Sender's name / address
    task_description: Plain-English description of what the agent should do
    reward          : Score awarded for the last action (0.0 on reset)
    done            : True when the episode has ended
    feedback        : Human-readable feedback string for learning
    score_breakdown : Dict with partial scores for each criterion
    """
    email_id: str
    subject: str
    body: str
    sender: str
    task_description: str
    reward: float
    done: bool
    feedback: str = ""
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class EmailState(State):
    """Episode metadata (extends core State with env-specific fields)."""
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.0
    difficulty: str = "easy"