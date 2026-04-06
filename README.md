---
title: Email Triage & Drafting Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📧 Email Triage & Drafting — OpenEnv Environment

A real-world **OpenEnv**-compliant RL environment where an AI agent learns to:
1. **Classify** incoming emails by priority (`urgent` / `normal` / `low`)
2. **Draft** professional, context-aware replies

Built for the [Scaler × Meta-PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

---

## 🎯 Why This Environment?

Email management is one of the most universal, high-stakes business tasks. Unlike toy environments, this one:
- Has **real-world grading criteria** (keyword coverage, tone, specificity)
- Rewards **partial progress** — no binary pass/fail
- Tests both **classification and generation** in a single reward signal
- Has clear **difficulty progression** from incident alerts to board-level communication

---

## 🧩 Environment Design

### Three Tasks (Easy → Hard)

| Task | Difficulty | Requires Reply | Max Reward |
|------|-----------|----------------|------------|
| Internal Server Incident Alert | Easy | ❌ | 1.0 |
| Enterprise Client Complaint | Medium | ✅ | 1.0 |
| Board Member Strategic Concerns | Hard | ✅ | 1.0 |

### Reward Function

Each task is scored out of **1.0** with **partial credit**:

| Component | Weight | Criteria |
|-----------|--------|----------|
| Priority Classification | 0.4 | Exact match to ground truth |
| Reply Quality | 0.6 | 70% keyword coverage + 30% length |

The reply score is continuous — a partial reply gets partial credit.

### Action Space

```json
{
  "priority": "urgent | normal | low",
  "reply_draft": "string (professional reply text)",
  "reasoning": "string (optional chain-of-thought)"
}
```

### Observation Space

```json
{
  "email_id": "string",
  "subject": "string",
  "body": "string",
  "sender": "string",
  "task_description": "string",
  "reward": "float (0.0-1.0)",
  "done": "boolean",
  "feedback": "string",
  "score_breakdown": {
    "priority_score": "float",
    "reply_score": "float",
    "total": "float"
  }
}
```

---

## 🚀 Quick Start

### Connect to the deployed environment

```python
from client import EmailTriageEnv
from models import EmailAction

with EmailTriageEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
    obs = env.reset()
    print(f"Email from {obs.sender}: {obs.subject}")

    result = env.step(EmailAction(
        priority="urgent",
        reply_draft="Dear Sarah, I sincerely apologise for the inconvenience...",
        reasoning="Enterprise client with explicit escalation request",
    ))
    print(f"Reward: {result.reward:.3f}")
    print(result.observation.feedback)
```

### Run the baseline inference script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="https://YOUR-SPACE.hf.space"

python inference.py
```

---

## 🛠 Local Setup

### Requirements
- Python 3.10+
- Docker
- `openenv-core`

### Run locally

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic websockets

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export ENV_BASE_URL="http://localhost:7860"
export HF_TOKEN="hf_..."
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

---

## 📁 Project Structure

```
email_triage_env/
├── inference.py          ← Baseline agent (root-level, required)
├── models.py             ← Type-safe Action / Observation / State
├── client.py             ← EnvClient subclass
├── openenv.yaml          ← OpenEnv spec
├── Dockerfile            ← Container definition
├── README.md             ← This file
└── server/
    ├── app.py            ← FastAPI entry-point
    ├── email_environment.py  ← Environment logic + graders
    └── requirements.txt
```

---

## 📊 Baseline Results

Running `Qwen/Qwen2.5-72B-Instruct` as the agent:

| Task | Priority Score | Reply Score | Total |
|------|---------------|-------------|-------|
| Easy — Incident Alert | 0.40 | 0.60 | **1.00** |
| Medium — Client Complaint | 0.40 | 0.52 | **0.92** |
| Hard — Board Escalation | 0.40 | 0.48 | **0.88** |
| **Cumulative** | | | **2.80 / 3.00** |

---

## 🔧 Configuration

Required environment variables for inference:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API token |
| `ENV_BASE_URL` | Environment server URL (default: http://localhost:7860) |