"""Simple async load test for the recruitment API.

Fires N requests at a target endpoint with a bounded concurrency and reports
latency percentiles, throughput, and the HTTP status distribution (so rate
limiting shows up as 429s).

Usage:
  python loadtest.py --url http://localhost:8000 --path /api/analytics/fairness \
      --concurrency 20 --requests 500
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

import aiohttp


async def _one(session, url, sem, latencies, statuses):
    async with sem:
        start = time.perf_counter()
        try:
            async with session.get(url) as response:
                await response.read()
                statuses[response.status] += 1
        except Exception as exc:  # noqa: BLE001
            statuses[f"err:{type(exc).__name__}"] += 1
        latencies.append((time.perf_counter() - start) * 1000.0)


async def run(base: str, path: str, concurrency: int, total: int) -> None:
    url = base.rstrip("/") + path
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: Counter = Counter()

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.perf_counter()
        await asyncio.gather(*[_one(session, url, sem, latencies, statuses) for _ in range(total)])
        wall = time.perf_counter() - wall_start

    latencies.sort()

    def pct(p: float) -> float:
        return round(latencies[min(len(latencies) - 1, int(len(latencies) * p))], 1)

    ok = sum(count for status, count in statuses.items() if isinstance(status, int) and 200 <= status < 400)
    print(f"target      : {url}")
    print(f"requests    : {total}  concurrency: {concurrency}")
    print(f"wall        : {round(wall, 2)}s")
    print(f"throughput  : {round(total / wall, 1)} req/s")
    print(f"success     : {ok}/{total}")
    print(f"status dist : {dict(statuses)}")
    print(f"latency ms  : p50={pct(0.5)} p90={pct(0.9)} p95={pct(0.95)} p99={pct(0.99)} max={round(latencies[-1], 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--path", default="/api/analytics/fairness")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.path, args.concurrency, args.requests))
