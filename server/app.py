"""
FastAPI application entry-point for the Email Triage environment.
"""

import os
# MANDATORY: Set environment variables before any other imports
os.environ["ENABLE_WEB_INTERFACE"] = "true"

import sys
import uvicorn

# Ensure the root directory is in the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core.env_server import create_app
from server.email_environment import EmailTriageEnvironment
from models import EmailAction, EmailObservation

# Create the FastAPI app
app = create_app(
    env=EmailTriageEnvironment,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
    env_name="email-triage-env",
    max_concurrent_envs=50,
)


def main():
    """Main entry point for the server CLI."""
    # Ensure reload=False for stable deployment in the validator
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()