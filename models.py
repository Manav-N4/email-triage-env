"""
Type definitions for the Email Triage environment.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from openenv.core.env_server.types import State


class EmailAction(BaseModel):
    priority: str = "normal"
    reply_draft: str = ""
    reasoning: str = ""


class EmailObservation(BaseModel):
    """
    State object returned by reset() and step().
    """
    email_id: str = ""
    subject: str = ""
    body: str = ""
    sender: str = ""
    task_description: str = ""
    
    # Strictly between 0 and 1
    reward: float = Field(default=0.33)
    done: bool = False
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    """
    Internal session state.
    """
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.33
    difficulty: str = "medium"