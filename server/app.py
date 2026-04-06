"""
FastAPI application entry-point for the Email Triage environment.
OpenEnv's create_app() wires up the WebSocket /ws endpoint,
the /health endpoint, and the Gradio web UI automatically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server import create_app
from server.email_environment import EmailTriageEnvironment

app = create_app(
    env_class=EmailTriageEnvironment,
    max_concurrent_envs=50,
    enable_web_interface=True,
)