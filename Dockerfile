FROM python:3.11-slim

LABEL maintainer="iaminov <iaminov@users.noreply.github.com>"
LABEL description="Production-ready ethical virtual therapist powered by Anthropic Claude"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml ./

RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "therapeutic_agent.api.main"]
