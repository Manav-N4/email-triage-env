"""
Email Triage & Drafting Environment — Type-safe models.

Action      : What the agent sends (priority label + reply draft)
Observation : What the agent receives (email content + feedback)
State       : Episode-level metadata
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core.env_server.types import State


class EmailAction(BaseModel):
    """
    The agent's response to a presented email.
    """
    priority: str = Field(description='One of "urgent", "normal", "low"')
    reply_draft: str = Field(description="The agent's suggested reply")
    reasoning: str = Field(default="", description="Optional chain-of-thought")


class EmailObservation(BaseModel):
    """
    What the environment sends back after reset() or step().
    """
    email_id: str
    subject: str
    body: str
    sender: str
    task_description: str
    reward: float = 0.0
    done: bool = False
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    """Episode metadata (extends core State with env-specific fields)."""
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.0
    difficulty: str = "easy"