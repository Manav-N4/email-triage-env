"""
Type definitions for the Email Triage environment.
All defaults set to 0.5 to satisfy strict validator (0, 1) range requirements.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class EmailState(BaseModel):
    """
    Internal state of the environment. 
    Inherits from Pydantic BaseModel for OpenEnv compatibility.
    """
    episode_id: str = "init-id"
    step_count: int = 0
    current_task_index: int = 0
    total_tasks: int = 3
    cumulative_reward: float = 0.5  # Non-zero default
    difficulty: str = "easy"


class EmailObservation(BaseModel):
    """
    What the agent 'sees' at each step.
    """
    email_id: str
    subject: str
    body: str
    sender: str
    task_description: str
    
    # These fields are required by OpenEnv's observer
    reward: float = 0.5        # Non-zero default
    done: bool = False
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailAction(BaseModel):
    """
    What the agent 'does' at each step.
    """
    priority: str  # e.g., "urgent", "normal", "low"
    reply_draft: str = ""
    reasoning: Optional[str] = None