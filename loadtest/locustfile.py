"""Locust load test — stress test the Kaizen RAG API.

Usage:
    pip install locust
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # Headless (CI-friendly):
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 --run-time 60s \
           --csv loadtest/results

Environment:
    API_KEY  — if auth is enabled, set this to a valid key
"""

import json
import os
import random

from locust import HttpUser, between, task

API_KEY = os.getenv("API_KEY", "")

# Sample queries — mix of simple, complex, and out-of-corpus
QUERIES = [
    "what is RAG?",
    "how do transformers work?",
    "compare GPT-4 and Claude",
    "best GPU for inference",
    "explain attention mechanism in neural networks",
    "kubernetes vs docker swarm for ML workloads",
    "how to fine-tune a language model with LoRA",
    "what is the difference between embedding and reranking?",
    "summarize the architecture of a vector database",
    "pip install torch failing on Windows",
    "SELECT * FROM users WHERE active = 1",
    "how to deploy FastAPI with Docker",
]

CATEGORIES = [None, "ml", "infra", "tools", "cloud"]


class KaizenUser(HttpUser):
    """Simulates a typical API consumer."""

    wait_time = between(0.5, 2.0)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    # -- Health / Status (lightweight, frequent) --

    @task(3)
    def health_check(self):
        self.client.get("/api/health", headers=self._headers(), name="/api/health")

    @task(1)
    def status_check(self):
        self.client.get("/api/status", headers=self._headers(), name="/api/status")

    @task(1)
    def metrics_check(self):
        self.client.get("/metrics", name="/metrics")

    # -- Query (main workload) --

    @task(10)
    def query_rag(self):
        payload = {
            "query": random.choice(QUERIES),
            "top_k": random.choice([3, 5, 7]),
            "category": random.choice(CATEGORIES),
        }
        with self.client.post(
            "/api/query",
            json=payload,
            headers=self._headers(),
            stream=True,
            catch_response=True,
            name="/api/query",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
                return

            # Consume SSE stream — verify we get sources + done events
            got_sources = False
            got_done = False
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                if text.startswith("data: "):
                    try:
                        event = json.loads(text[6:])
                        if event.get("type") == "sources":
                            got_sources = True
                        elif event.get("type") == "done":
                            got_done = True
                    except json.JSONDecodeError:
                        pass

            if not got_sources or not got_done:
                resp.failure("Missing SSE events (sources/done)")
            else:
                resp.success()

    # -- Gaps (read-heavy) --

    @task(1)
    def gaps_check(self):
        self.client.get("/api/gaps", headers=self._headers(), name="/api/gaps")


class HighThroughputUser(HttpUser):
    """Aggressive user — shorter wait, health-only. For burst testing."""

    wait_time = between(0.1, 0.3)

    def _headers(self) -> dict:
        h = {}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    @task
    def burst_health(self):
        self.client.get("/api/health", headers=self._headers(), name="/api/health [burst]")
