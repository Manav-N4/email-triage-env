"""
EmailTriageEnv client — import this in your training or inference code.

Usage (sync):
    from client import EmailTriageEnv
    from models import EmailAction

    with EmailTriageEnv(base_url="https://YOUR-HF-SPACE.hf.space").sync() as env:
        obs = env.reset()
        print(obs.subject)

        result = env.step(EmailAction(
            priority="urgent",
            reply_draft="Dear Sarah, we sincerely apologise...",
        ))
        print(result.reward)

Usage (async):
    async with EmailTriageEnv(base_url="...") as env:
        obs = await env.reset()
        result = await env.step(EmailAction(...))
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import EmailAction, EmailObservation, EmailState


class EmailTriageEnv(EnvClient[EmailAction, EmailObservation, EmailState]):
    """
    WebSocket client for the Email Triage & Drafting environment.
    Maintains a persistent connection; each step() is a lightweight frame.
    """

    def _step_payload(self, action: EmailAction) -> dict:
        return {
            "priority": action.priority,
            "reply_draft": action.reply_draft,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult[EmailObservation]:
        obs_data = payload.get("observation", {})
        obs = EmailObservation(
            email_id=obs_data.get("email_id", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            task_description=obs_data.get("task_description", ""),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> EmailState:
        return EmailState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_index=payload.get("current_task_index", 0),
            total_tasks=payload.get("total_tasks", 3),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            difficulty=payload.get("difficulty", "easy"),
        )