"""
Email Triage & Drafting Environment — Pydantic models.
Defaults set to 0.5 (neutral mid-point) for headroom.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core.env_server.types import State


class EmailAction(BaseModel):
    priority: str = "normal"
    reply_draft: str = ""
    reasoning: str = ""


class EmailObservation(BaseModel):
    email_id: str = ""
    subject: str = ""
    body: str = ""
    sender: str = ""
    task_description: str = ""
    # IMPORTANT: Mid-interval neutral default
    reward: float = Field(default=0.5)
    done: bool = False
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.5   # Neutral start
    difficulty: str = "medium"