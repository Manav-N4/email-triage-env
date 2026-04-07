FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using the system-wide python
# This ensures all our requirements from pyproject.toml are met
RUN uv pip install --system .

# Copy the rest of the application
COPY . .

# Ensure the root is in the python path
ENV PYTHONPATH=/app

# Expose the default OpenEnv/Gradio port
EXPOSE 7860

# Run the server via our standardized main() entry point
CMD ["python", "-m", "server.app"]