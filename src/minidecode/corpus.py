"""Training text: synthetic web server access logs.

Logs suit a tiny model. They have real structure (a clock that only moves
forward, methods that go with certain paths, status codes and latencies that
depend on the endpoint), the alphabet is small, and a generated line can be
checked mechanically, which gives a quality measure beyond perplexity.

    09:14:03 GET /api/orders/4821 200 38ms
"""

import random
import re

# path template -> (methods, typical latency in ms)
ENDPOINTS = {
    "/": (("GET",), 12),
    "/health": (("GET",), 2),
    "/login": (("GET", "POST"), 85),
    "/api/orders": (("GET", "POST"), 60),
    "/api/orders/{id}": (("GET", "PUT", "DELETE"), 35),
    "/api/users/{id}": (("GET", "PUT"), 25),
    "/api/search": (("GET",), 140),
    "/static/app.js": (("GET",), 5),
}

OK = {"GET": 200, "POST": 201, "PUT": 200, "DELETE": 204}
ERRORS = (400, 401, 404, 500, 503)


def line(rng: random.Random, seconds: int) -> str:
    path = rng.choice(list(ENDPOINTS))
    methods, typical = ENDPOINTS[path]
    method = rng.choice(methods)
    path = path.replace("{id}", str(rng.randint(1, 9999)))
    status = OK[method] if rng.random() > 0.08 else rng.choice(ERRORS)
    ms = max(1, round(rng.lognormvariate(0, 0.5) * typical))
    h, m, s = seconds // 3600 % 24, seconds // 60 % 60, seconds % 60
    return f"{h:02}:{m:02}:{s:02} {method} {path} {status} {ms}ms"


def generate(lines: int, seed: int = 0, block: int = 200) -> str:
    """`lines` log lines. Every `block` lines the clock jumps to a random time
    of day and moves forward from there, so any stretch of the text covers
    the whole day. (One clock running through would put hours in the
    validation split that training never saw.)"""
    rng = random.Random(seed)
    out = []
    for i in range(lines):
        if i % block == 0:
            seconds = rng.randint(0, 86399)
        seconds += rng.choice((0, 0, 1, 1, 2, 3))
        out.append(line(rng, seconds))
    return "\n".join(out) + "\n"


LINE = re.compile(r"(\d\d):(\d\d):(\d\d) ([A-Z]+) (\S+) (\d{3}) (\d+)ms")


def _template(path: str) -> str | None:
    if path in ENDPOINTS:
        return path
    head, _, last = path.rpartition("/")
    if last.isdigit() and 1 <= int(last) <= 9999 and f"{head}/{{id}}" in ENDPOINTS:
        return f"{head}/{{id}}"
    return None


def problem(line: str) -> str | None:
    """Why a line could not have come from the generator, or None if it could."""
    m = LINE.fullmatch(line)
    if not m:
        return "not a log line"
    h, mi, s, method, path, status, _ = m.groups()
    if int(h) > 23 or int(mi) > 59 or int(s) > 59:
        return "bad time"
    template = _template(path)
    if template is None:
        return "unknown path"
    if method not in ENDPOINTS[template][0]:
        return f"{method} not used on {template}"
    if int(status) not in (OK[method], *ERRORS):
        return f"status {status} after {method}"
    return None


def valid_rate(text: str) -> float:
    """Share of complete lines (ending in a newline) that pass `problem`."""
    lines = text.split("\n")[:-1]
    if not lines:
        return 0.0
    return sum(problem(x) is None for x in lines) / len(lines)
