#!/usr/bin/env python3
"""
Aria Monitor — lightweight vLLM dashboard for DGX Spark.
Zero external dependencies. Python stdlib + Chart.js CDN.

Polls vLLM /metrics (Prometheus exposition) + nvidia-smi,
serves a real-time dashboard with cost-savings comparison.
"""

import collections
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config via env vars
# ---------------------------------------------------------------------------
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
API_KEY = os.environ.get("API_KEY", "")
MONITOR_PORT = int(os.environ.get("MONITOR_PORT", "8090"))
GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1"))
HISTORY_MAX = int(os.environ.get("HISTORY_MAX", "300"))          # 5min @ 1s
HISTORY_HOUR_MAX = int(os.environ.get("HISTORY_HOUR_MAX", "3600"))  # 1h @ 1s
HISTORY_24H_MAX = int(os.environ.get("HISTORY_24H_MAX", "86400"))   # 24h @ 1s

MODEL_NAME = os.environ.get("MODEL_NAME", "aria-27b")
MODEL_DISPLAY = os.environ.get("MODEL_DISPLAY", "Aria 27B (NVFP4)")

ZAR_PER_KWH = float(os.environ.get("ZAR_PER_KWH", "2"))  # SA electricity rate
USD_TO_ZAR = float(os.environ.get("USD_TO_ZAR", "16.52"))  # exchange rate

# Cost per 1M tokens — frontier model pricing (USD, as of August 2026)
# Best-case pricing: using introductory/promotional rates where available.
# Cache read rates included for fair comparison against local prefix caching.
# Claude Sonnet 5: $2/$10 introductory through Aug 31 2026 (standard $3/$15)
# Claude Opus 5: $5/$25
# Claude cache read: 10% of base input
# GPT-5.6 Sol: $5/$30 (flagship)
# GPT-5.6 Terra: $2.50/$15 (balanced)
# GPT-5.6 Luna: $1/$6 (fast, low-cost)
# GPT-5.6 cache read: 90% off uncached input
COST_TABLE: dict[str, dict[str, float]] = {
    "Aria 27B (Local)":    {"input": 0.0,   "output": 0.0, "cached_input": 0.0},
    "Claude Sonnet 5":     {"input": 2.00,  "output": 10.00, "cached_input": 0.20},
    "Claude Opus 5":       {"input": 5.00,  "output": 25.00, "cached_input": 0.50},
    "Claude Fable 5":      {"input": 10.00, "output": 50.00, "cached_input": 1.00},
    "GPT-5.6 Sol":         {"input": 5.00,  "output": 30.00, "cached_input": 0.50},
    "GPT-5.6 Terra":       {"input": 2.50,  "output": 15.00, "cached_input": 0.25},
    "GPT-5.6 Luna":        {"input": 1.00,  "output": 6.00, "cached_input": 0.10},
}

# Subscription plans — effective per-token rates based on monthly fee and
# typical usage. These give a "best case" cost for heavy subscription users.
# Effective rate = monthly_price / estimated_monthly_tokens (split 1:3 input:output).
# ESTIMATED_MONTHLY_TOKENS controls the usage assumption (default 10M/month).
ESTIMATED_MONTHLY_TOKENS = float(os.environ.get("ESTIMATED_MONTHLY_TOKENS", "10"))  # millions

SUBSCRIPTION_PLANS: dict[str, dict] = {
    "ChatGPT Plus ($20/mo)": {
        "monthly_usd": 20,
        "models": ["GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna"],
    },
    "ChatGPT Pro ($200/mo)": {
        "monthly_usd": 200,
        "models": ["GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna"],
    },
    "Claude Pro ($20/mo)": {
        "monthly_usd": 20,
        "models": ["Claude Sonnet 5", "Claude Opus 5", "Claude Fable 5"],
    },
    "Claude Max ($200/mo)": {
        "monthly_usd": 200,
        "models": ["Claude Sonnet 5", "Claude Opus 5", "Claude Fable 5"],
    },
}


def compute_subscription_costs(total_prompt: int, total_gen: int) -> dict:
    """Compute effective per-token costs under subscription plans.
    
    Distributes the monthly fee across models based on estimated usage,
    then calculates effective $/token for the actual tokens processed.
    """
    total_tokens_m = ESTIMATED_MONTHLY_TOKENS  # millions per month
    results = {}
    for plan_name, plan in SUBSCRIPTION_PLANS.items():
        monthly = plan["monthly_usd"]
        models = plan["models"]
        # Effective rate per 1M tokens (total, not split input/output)
        # Assume 1:3 input:output token ratio typical
        effective_per_m = monthly / total_tokens_m
        # Split: 25% input, 75% output weighting
        effective_input_per_m = effective_per_m * 0.25
        effective_output_per_m = effective_per_m * 0.75

        for model_name in models:
            key = f"{model_name} ({plan_name})"
            input_cost = (total_prompt / 1_000_000) * effective_input_per_m
            output_cost = (total_gen / 1_000_000) * effective_output_per_m
            results[key] = {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(input_cost + output_cost, 4),
                "plan": plan_name,
                "monthly_usd": monthly,
                "effective_per_m": round(effective_per_m, 2),
                "model": model_name,
            }
    return results

VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/metrics"

# ---------------------------------------------------------------------------
# Prometheus metric parser
# ---------------------------------------------------------------------------
METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+(.+)$')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_prometheus(text: str) -> dict:
    """Parse Prometheus exposition format into {metric_name: [(labels_dict, value)]}."""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = METRIC_RE.match(line)
        if m:
            name = m.group(1)
            labels_str = m.group(2)
            value = float(m.group(3))
            labels = dict(LABEL_RE.findall(labels_str))
            metrics.setdefault(name, []).append((labels, value))
        else:
            # Simple "name value" line (no labels)
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    metrics.setdefault(parts[0], []).append(({}, float(parts[1])))
                except ValueError:
                    pass
    return metrics


def get_metric_value(metrics: dict, name: str, label_filter: dict = None) -> float:
    """Get a single metric value, optionally filtered by labels."""
    entries = metrics.get(name, [])
    for labels, value in entries:
        if label_filter:
            if all(labels.get(k) == v for k, v in label_filter.items()):
                return value
        else:
            return value
    return 0.0


# ---------------------------------------------------------------------------
# GPU stats via nvidia-smi
# ---------------------------------------------------------------------------
def get_gpu_stats() -> dict:
    """Query nvidia-smi for GPU temperature, utilization, VRAM, power."""
    try:
        def safe_float(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return 0.0

        cmd = [
            "nvidia-smi",
            f"--id={GPU_INDEX}",
            "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 7:
            stats = {
                "temp": safe_float(parts[0]),
                "gpu_util": safe_float(parts[1]),
                "mem_util": safe_float(parts[2]),
                "vram_used_mb": safe_float(parts[3]),
                "vram_total_mb": safe_float(parts[4]),
                "power_w": safe_float(parts[5]),
                "power_limit_w": safe_float(parts[6]),
            }
            # GB10 Blackwell doesn't report memory via --query-gpu.
            # Fallback: sum used_memory from compute apps.
            if stats["vram_used_mb"] == 0:
                try:
                    apps = subprocess.run(
                        ["nvidia-smi",
                         "--query-compute-apps=used_memory",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    total = 0
                    for line in apps.stdout.strip().splitlines():
                        total += safe_float(line.strip())
                    if total > 0:
                        stats["vram_used_mb"] = total
                except Exception:
                    pass
            return stats
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Snapshot collector
# ---------------------------------------------------------------------------
class Snapshot:
    """One point-in-time sample of all metrics."""
    __slots__ = (
        "ts", "prompt_tokens", "generation_tokens",
        "ttft_sum", "ttft_count", "e2e_sum", "e2e_count",
        "itl_sum", "itl_count",
        "requests_running", "requests_waiting", "kv_cache_usage",
        "request_success_stop", "request_success_error", "request_success_abort",
        "request_success_length",
        "spec_drafts", "spec_draft_tokens", "spec_accepted_tokens",
        "prefix_cache_hits", "prefix_cache_queries",
        "gpu",
    )

    def __init__(self, metrics: dict, gpu: dict):
        now = time.time()
        self.ts = now
        # Cumulative counters
        self.prompt_tokens = get_metric_value(metrics, "vllm:prompt_tokens_total")
        self.generation_tokens = get_metric_value(metrics, "vllm:generation_tokens_total")
        # Latency sums/counts
        self.ttft_sum = get_metric_value(metrics, "vllm:time_to_first_token_seconds_sum")
        self.ttft_count = get_metric_value(metrics, "vllm:time_to_first_token_seconds_count")
        self.e2e_sum = get_metric_value(metrics, "vllm:e2e_request_latency_seconds_sum")
        self.e2e_count = get_metric_value(metrics, "vllm:e2e_request_latency_seconds_count")
        self.itl_sum = get_metric_value(metrics, "vllm:inter_token_latency_seconds_sum")
        self.itl_count = get_metric_value(metrics, "vllm:inter_token_latency_seconds_count")
        # Gauges
        self.requests_running = get_metric_value(metrics, "vllm:num_requests_running")
        self.requests_waiting = get_metric_value(metrics, "vllm:num_requests_waiting")
        self.kv_cache_usage = get_metric_value(metrics, "vllm:kv_cache_usage_perc")
        # Request outcomes
        self.request_success_stop = get_metric_value(
            metrics, "vllm:request_success_total",
            {"finished_reason": "stop"})
        self.request_success_error = get_metric_value(
            metrics, "vllm:request_success_total",
            {"finished_reason": "error"})
        self.request_success_abort = get_metric_value(
            metrics, "vllm:request_success_total",
            {"finished_reason": "abort"})
        self.request_success_length = get_metric_value(
            metrics, "vllm:request_success_total",
            {"finished_reason": "length"})
        # Speculative decoding
        self.spec_drafts = get_metric_value(metrics, "vllm:spec_decode_num_drafts_total")
        self.spec_draft_tokens = get_metric_value(metrics, "vllm:spec_decode_num_draft_tokens_total")
        self.spec_accepted_tokens = get_metric_value(metrics, "vllm:spec_decode_num_accepted_tokens_total")
        # Prefix cache
        self.prefix_cache_hits = get_metric_value(metrics, "vllm:prefix_cache_hits_total")
        self.prefix_cache_queries = get_metric_value(metrics, "vllm:prefix_cache_queries_total")
        # GPU
        self.gpu = gpu


def compute_delta(prev: Snapshot, curr: Snapshot) -> dict:
    """Compute delta/instant metrics between two snapshots."""
    dt = curr.ts - prev.ts
    if dt <= 0:
        dt = 1.0

    delta_prompt = curr.prompt_tokens - prev.prompt_tokens
    delta_gen = curr.generation_tokens - prev.generation_tokens

    # Instant throughput (tokens/sec)
    prompt_tps = delta_prompt / dt
    gen_tps = delta_gen / dt

    # Request completions (diff of cumulative success counters)
    requests_completed = (
        (curr.request_success_stop - prev.request_success_stop)
        + (curr.request_success_error - prev.request_success_error)
        + (curr.request_success_abort - prev.request_success_abort)
        + (curr.request_success_length - prev.request_success_length)
    )

    # Spec decode acceptance rate
    spec_rate = 0.0
    if curr.spec_draft_tokens > 0:
        spec_rate = curr.spec_accepted_tokens / curr.spec_draft_tokens

    # Prefix cache hit rate
    cache_hit_rate = 0.0
    if curr.prefix_cache_queries > 0:
        cache_hit_rate = curr.prefix_cache_hits / curr.prefix_cache_queries

    # Latency averages (lifetime, from cumulative sum/count)
    avg_ttft = (curr.ttft_sum / curr.ttft_count) if curr.ttft_count > 0 else 0
    avg_e2e = (curr.e2e_sum / curr.e2e_count) if curr.e2e_count > 0 else 0
    avg_itl = (curr.itl_sum / curr.itl_count) if curr.itl_count > 0 else 0

    busy = curr.requests_running > 0 or delta_gen > 0

    return {
        "ts": curr.ts,
        "dt": dt,
        "delta_prompt_tokens": delta_prompt,
        "delta_gen_tokens": delta_gen,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "requests_running": curr.requests_running,
        "requests_waiting": curr.requests_waiting,
        "kv_cache_usage": curr.kv_cache_usage,
        "requests_completed": requests_completed,
        "busy": busy,
        # Cumulative totals
        "total_prompt_tokens": curr.prompt_tokens,
        "total_gen_tokens": curr.generation_tokens,
        "total_requests": int(curr.request_success_stop + curr.request_success_error
                             + curr.request_success_abort + curr.request_success_length),
        # Averages
        "avg_ttft_ms": avg_ttft * 1000,
        "avg_e2e_s": avg_e2e,
        "avg_itl_ms": avg_itl * 1000,
        # Spec decode
        "spec_acceptance_rate": spec_rate,
        "spec_drafts_total": curr.spec_drafts,
        # Cache
        "cache_hit_rate": cache_hit_rate,
        # GPU
        "gpu": curr.gpu,
    }


# ---------------------------------------------------------------------------
# Cost savings calculator
# ---------------------------------------------------------------------------
def compute_cost_savings(total_prompt: int, total_gen: int, cache_hit_rate: float = 0.0) -> dict:
    """Compare local ($0) cost against frontier model pricing.
    
    Applies the same cache hit rate to frontier models for fair comparison.
    If the local model has X% prefix cache hits, frontier models would also
    benefit from their caching on the same workload.
    """
    results = {}
    for model, prices in COST_TABLE.items():
        base_input = prices.get("input", 0.0)
        cached_input = prices.get("cached_input", base_input)
        output_price = prices.get("output", 0.0)

        # Input cost: uncached portion at full rate + cached portion at cache rate
        uncached_prompt = total_prompt * (1 - cache_hit_rate)
        cached_prompt = total_prompt * cache_hit_rate
        input_cost = (uncached_prompt / 1_000_000) * base_input + (cached_prompt / 1_000_000) * cached_input
        output_cost = (total_gen / 1_000_000) * output_price
        total = input_cost + output_cost
        results[model] = {
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(total, 4),
        }
    return results


# ---------------------------------------------------------------------------
# Collector loop with triple-deque architecture
# ---------------------------------------------------------------------------
history = collections.deque(maxlen=HISTORY_MAX)
history_hour = collections.deque(maxlen=HISTORY_HOUR_MAX)
history_24h = collections.deque(maxlen=HISTORY_24H_MAX)

prev_snapshot = None
latest_delta = {}
latest_gpu = {}
lock = threading.Lock()
start_time = time.time()
energy_wh_total = 0.0  # cumulative GPU energy in watt-hours


def fetch_vllm_metrics() -> dict:
    """Fetch and parse vLLM Prometheus metrics."""
    req = urllib.request.Request(VLLM_URL)
    if API_KEY:
        req.add_header("Authorization", f"Bearer {API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return parse_prometheus(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[aria-monitor] Failed to fetch vLLM metrics: {e}")
        return {}


def collector_loop():
    """Background daemon: poll vLLM + GPU, compute deltas, store in deques."""
    global prev_snapshot, latest_delta, latest_gpu

    # Wait for vLLM to be available
    print(f"[aria-monitor] Waiting for vLLM at {VLLM_URL}...")
    while True:
        metrics = fetch_vllm_metrics()
        if metrics:
            print(f"[aria-monitor] Connected to vLLM. Starting collection.")
            break
        time.sleep(5)

    while True:
        try:
            metrics = fetch_vllm_metrics()
            gpu = get_gpu_stats()

            if metrics:
                snap = Snapshot(metrics, gpu)

                with lock:
                    global energy_wh_total
                    latest_gpu = gpu

                    # Accumulate GPU energy (watt-hours)
                    if prev_snapshot is not None:
                        dt_hours = (snap.ts - prev_snapshot.ts) / 3600.0
                        power_w = gpu.get("power_w", 0)
                        if power_w > 0:
                            energy_wh_total += power_w * dt_hours

                    if prev_snapshot is not None:
                        delta = compute_delta(prev_snapshot, snap)
                        latest_delta = delta
                        history.append(delta)
                        history_hour.append(delta)
                        history_24h.append(delta)
                    else:
                        # First snapshot — seed with zeros
                        delta = {
                            "ts": snap.ts, "dt": 0,
                            "delta_prompt_tokens": 0, "delta_gen_tokens": 0,
                            "prompt_tps": 0, "gen_tps": 0,
                            "requests_running": snap.requests_running,
                            "requests_waiting": snap.requests_waiting,
                            "kv_cache_usage": snap.kv_cache_usage,
                            "requests_completed": 0, "busy": False,
                            "total_prompt_tokens": snap.prompt_tokens,
                            "total_gen_tokens": snap.generation_tokens,
                            "total_requests": 0,
                            "avg_ttft_ms": 0, "avg_e2e_s": 0, "avg_itl_ms": 0,
                            "spec_acceptance_rate": 0, "spec_drafts_total": 0,
                            "cache_hit_rate": 0,
                            "gpu": gpu,
                        }
                        latest_delta = delta

                    prev_snapshot = snap

        except Exception as e:
            print(f"[aria-monitor] Collector error: {e}")

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Rolling averages (active-only) from hourly deque
# ---------------------------------------------------------------------------
def compute_hourly_averages() -> dict:
    """Compute active-only rolling averages from the hourly deque."""
    with lock:
        entries = list(history_hour)

    if not entries:
        return {
            "avg_gen_tps_1h": 0, "avg_prompt_tps_1h": 0,
            "avg_ttft_ms_1h": 0, "gen_samples_1h": 0,
            "prompt_samples_1h": 0,
        }

    gen_vals = [e["gen_tps"] for e in entries if e["gen_tps"] > 0]
    prompt_vals = [e["prompt_tps"] for e in entries if e["prompt_tps"] > 0]
    ttft_vals = [e["avg_ttft_ms"] for e in entries if e["avg_ttft_ms"] > 0]

    return {
        "avg_gen_tps_1h": round(sum(gen_vals) / len(gen_vals), 1) if gen_vals else 0,
        "avg_prompt_tps_1h": round(sum(prompt_vals) / len(prompt_vals), 1) if prompt_vals else 0,
        "avg_ttft_ms_1h": round(sum(ttft_vals) / len(ttft_vals), 1) if ttft_vals else 0,
        "gen_samples_1h": len(gen_vals),
        "prompt_samples_1h": len(prompt_vals),
    }


def compute_24h_totals() -> dict:
    """Compute 24h rolling token/request totals from deque counter diffs."""
    with lock:
        entries = list(history_24h)

    if len(entries) < 2:
        return {"tokens_gen_24h": 0, "tokens_prompt_24h": 0, "requests_24h": 0}

    oldest = entries[0]
    newest = entries[-1]

    return {
        "tokens_gen_24h": max(0, newest["total_gen_tokens"] - oldest["total_gen_tokens"]),
        "tokens_prompt_24h": max(0, newest["total_prompt_tokens"] - oldest["total_prompt_tokens"]),
        "requests_24h": max(0, newest["total_requests"] - oldest["total_requests"]),
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class MonitorHandler(BaseHTTPRequestHandler):
    """Serve dashboard HTML and JSON API."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_json(self._get_status())
        elif self.path == "/api/history":
            self._serve_json(self._get_history())
        else:
            self.send_error(404)

    def _serve_html(self):
        html_path = os.path.join(os.path.dirname(__file__) or ".", "dashboard.html")
        try:
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(500, "dashboard.html not found")

    def _serve_json(self, data):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _get_status(self) -> dict:
        with lock:
            delta = dict(latest_delta) if latest_delta else {}
            gpu = dict(latest_gpu) if latest_gpu else {}

        hourly = compute_hourly_averages()
        daily = compute_24h_totals()

        total_prompt = delta.get("total_prompt_tokens", 0)
        total_gen = delta.get("total_gen_tokens", 0)
        total_requests = delta.get("total_requests", 0)
        cache_hit_rate = delta.get("cache_hit_rate", 0)
        cost_savings = compute_cost_savings(total_prompt, total_gen, cache_hit_rate)
        sub_costs = compute_subscription_costs(total_prompt, total_gen)

        # 24h cost savings
        daily_prompt = daily.get("tokens_prompt_24h", 0)
        daily_gen = daily.get("tokens_gen_24h", 0)
        cost_savings_24h = compute_cost_savings(daily_prompt, daily_gen, cache_hit_rate)
        sub_costs_24h = compute_subscription_costs(daily_prompt, daily_gen)

        uptime_s = time.time() - start_time
        start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%Y-%m-%d")

        # Electricity cost
        with lock:
            wh_total = energy_wh_total
        # Estimate 24h energy from current power draw * 24h (approximation)
        # More accurate would be to track a 24h energy deque, but this is reasonable
        power_now = gpu.get("power_w", 0)
        wh_24h = daily.get("tokens_prompt_24h", 0)  # placeholder, use uptime ratio
        # Use uptime-based ratio for 24h estimate
        if uptime_s > 0:
            wh_24h = wh_total * min(86400 / uptime_s, 1.0)

        electricity = {
            "zar_per_kwh": ZAR_PER_KWH,
            "total_wh": round(wh_total, 2),
            "total_kwh": round(wh_total / 1000, 4),
            "total_cost_zar": round((wh_total / 1000) * ZAR_PER_KWH, 2),
            "total_cost_usd": round(((wh_total / 1000) * ZAR_PER_KWH) / USD_TO_ZAR, 4),
            "est_24h_wh": round(wh_24h, 2),
            "est_24h_cost_zar": round((wh_24h / 1000) * ZAR_PER_KWH, 2),
            "est_24h_cost_usd": round(((wh_24h / 1000) * ZAR_PER_KWH) / USD_TO_ZAR, 4),
        }

        return {
            "now": time.time(),
            "uptime_s": uptime_s,
            "started": start_iso,
            "model": MODEL_DISPLAY,
            "vllm_host": VLLM_HOST,
            "vllm_port": VLLM_PORT,
            "instant": delta,
            "hourly": hourly,
            "daily": daily,
            "cost_savings": cost_savings,
            "cost_savings_24h": cost_savings_24h,
            "subscription_costs": sub_costs,
            "subscription_costs_24h": sub_costs_24h,
            "cache_hit_rate": cache_hit_rate,
            "electricity": electricity,
            "cost_table": COST_TABLE,
        }

    def _get_history(self) -> list:
        with lock:
            return list(history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Start collector thread
    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", MONITOR_PORT), MonitorHandler)
    server.socket.setsockopt(
        __import__("socket").SOL_SOCKET,
        __import__("socket").SO_REUSEADDR, 1
    )
    print(f"[aria-monitor] Dashboard at http://0.0.0.0:{MONITOR_PORT}")
    print(f"[aria-monitor] Monitoring vLLM at {VLLM_URL}")
    print(f"[aria-monitor] Model: {MODEL_DISPLAY}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[aria-monitor] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
