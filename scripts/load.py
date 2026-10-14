"""Fire requests at a running `minidecode serve` and report throughput.

uv run minidecode serve runs/model.npz &
uv run python scripts/load.py --requests 200 --concurrency 1 8 32
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request


def one(url: str) -> tuple[float, int]:
    body = json.dumps({"prompt": "\n", "max_new": 46, "lines": 1}).encode()
    start = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, body, method="POST")) as r:
        tokens = json.loads(r.read())["tokens"]
    return time.perf_counter() - start, tokens


def run(url: str, requests: int, concurrency: int) -> dict:
    lock = threading.Lock()
    left = [requests]
    results: list[tuple[float, int]] = []

    def client():
        while True:
            with lock:
                if not left[0]:
                    return
                left[0] -= 1
            r = one(url)
            with lock:
                results.append(r)

    start = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = time.perf_counter() - start
    lat = sorted(s for s, _ in results)
    return {
        "concurrency": concurrency,
        "seconds": seconds,
        "requests_per_s": len(results) / seconds,
        "tokens_per_s": sum(n for _, n in results) / seconds,
        "p50_ms": statistics.median(lat) * 1000,
        "p95_ms": lat[int(0.95 * (len(lat) - 1))] * 1000,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/generate")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    args = p.parse_args()
    run(args.url, 10, 2)  # warm up
    print(f"{'clients':>8} {'req/s':>8} {'tokens/s':>9} {'p50':>8} {'p95':>8}")
    for c in args.concurrency:
        r = run(args.url, args.requests, c)
        print(
            f"{c:8} {r['requests_per_s']:8.1f} {r['tokens_per_s']:9.0f}"
            f" {r['p50_ms']:6.0f}ms {r['p95_ms']:6.0f}ms"
        )


if __name__ == "__main__":
    main()
