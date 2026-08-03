# Aria Monitor

Lightweight real-time dashboard for monitoring vLLM on the DGX Spark.

Zero external Python dependencies — uses stdlib `http.server` + Chart.js CDN.

## Features

- Real-time token throughput (generation + prompt) with 1s polling
- GPU stats via nvidia-smi (temp, utilization, VRAM, power)
- KV cache usage, prefix cache hit rate, speculative decode acceptance
- Latency tracking (TTFT, E2E, inter-token)
- **Cost savings comparison** against frontier models (GPT-4o, Claude, Gemini)
- Rolling averages (1h active-only) and 24h token totals
- Dark theme, auto-refreshing Chart.js charts

## Quick Start

```bash
# Run directly
python3 monitor.py

# Open http://localhost:8090
```

## Environment Variables

| Variable        | Default       | Description                      |
|-----------------|---------------|----------------------------------|
| VLLM_HOST       | localhost     | vLLM server host                 |
| VLLM_PORT       | 8000          | vLLM server port                 |
| API_KEY         | (empty)       | Bearer token for vLLM metrics    |
| MONITOR_PORT    | 8090          | Dashboard HTTP port              |
| GPU_INDEX       | 0             | nvidia-smi GPU index             |
| POLL_INTERVAL   | 1             | Seconds between polls            |
| MODEL_NAME      | aria-27b      | vLLM model name for metrics      |
| MODEL_DISPLAY   | Aria 27B (NVFP4) | Display name in dashboard     |

## Deploy as systemd Service

```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp aria-monitor.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now aria-monitor

# Check status
systemctl --user status aria-monitor
```

## Cost Table

Update frontier model pricing in `monitor.py` (`COST_TABLE` dict) as prices change.
Local model cost is always $0 — the dashboard calculates what the same tokens
would have cost on each frontier model.

## Architecture

```
monitor.py (HTTP server, port 8090)
  ├── Background thread (1s poll)
  │     ├── Fetch vLLM /metrics (Prometheus exposition)
  │     ├── Parse vllm:* counters + gauges
  │     ├── Query nvidia-smi for GPU stats
  │     ├── Compute deltas (instant throughput)
  │     └── Store in triple deques (5min / 1h / 24h)
  ├── /api/status  → latest snapshot + averages + cost savings
  ├── /api/history → chart data (5 min window)
  └── /            → dashboard.html (Chart.js)
```
