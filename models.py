"""
Email Triage & Drafting Environment — Pydantic models.

All reward fields default to 0.01 (never 0.0) to satisfy the OpenEnv
validator requirement that every reward value is strictly in (0, 1).
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core.env_server.types import State


class EmailAction(BaseModel):
    priority: str = "normal"          # "urgent" | "normal" | "low"
    reply_draft: str = ""
    reasoning: str = ""


class EmailObservation(BaseModel):
    email_id: str = ""
    subject: str = ""
    body: str = ""
    sender: str = ""
    task_description: str = ""
    # IMPORTANT: Never default to 0.0 — validator checks every emitted reward
    reward: float = Field(default=0.01)
    done: bool = False
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.01   # Never 0.0
    difficulty: str = "medium"