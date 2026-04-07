"""
FastAPI application entry-point for the Email Triage environment.
OpenEnv's create_app() wires up the WebSocket /ws endpoint,
the /health endpoint, and the Gradio web UI automatically.
"""

import sys, os
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server import create_app
from server.email_environment import EmailTriageEnvironment
from models import EmailAction, EmailObservation

os.environ["ENABLE_WEB_INTERFACE"] = "true"

app = create_app(
    env=EmailTriageEnvironment,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
    env_name="email-triage-env",
    max_concurrent_envs=50,
)


def main():
    """Main entry point for the server CLI."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()