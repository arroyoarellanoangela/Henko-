# Load Testing

Stress test the Kaizen RAG API with [Locust](https://locust.io/).

## Setup

```bash
pip install locust
```

## Usage

### Interactive (web UI)
```bash
locust -f loadtest/locustfile.py --host http://localhost:8000
# Open http://localhost:8089 in browser
```

### Headless (CI)
```bash
# 50 users, ramp 5/sec, run 60 seconds
locust -f loadtest/locustfile.py --host http://localhost:8000 \
       --headless -u 50 -r 5 --run-time 60s \
       --csv loadtest/results
```

### With auth
```bash
API_KEY=your-key locust -f loadtest/locustfile.py --host http://localhost:8000
```

## User Types

- **KaizenUser**: Realistic mix — 60% queries, 20% health, 10% status/gaps/metrics
- **HighThroughputUser**: Burst mode — rapid health checks to test rate limiting

## Output

CSV results in `loadtest/results_*.csv` (requests, failures, response times).
