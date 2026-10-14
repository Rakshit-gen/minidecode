"""An HTTP front end for the engine, stdlib only.

Handler threads put requests on the engine and wait. One worker thread owns
the engine and steps it while there is work, so requests that arrive while
others are decoding join the running batch instead of waiting for it.

    POST /generate  {"prompt": "12:0", "max_new": 40, "temperature": 0.8, "lines": 1}
    GET  /health
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .prefix import PrefixCache
from .serve import Engine, Request

MAX_BODY = 64 * 1024  # a prompt can't be longer than the context anyway


class Worker:
    def __init__(self, params, cfg, tok, slots=8, temperature=0.0, seed=0):
        self.cfg, self.tok = cfg, tok
        self.engine = Engine(
            params, cfg, slots, temperature=temperature, seed=seed, prefix=PrefixCache(cfg)
        )
        self.cond = threading.Condition()
        self.done: dict[int, threading.Event] = {}
        self.next_id = 0
        self.served = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: str, max_new: int, temperature=None, lines=0) -> Request:
        """Queue a request and block until it finishes. Raises ValueError for
        requests the model can't take."""
        ids = list(self.tok.encode(prompt))
        newline = int(self.tok.encode("\n")[0]) if lines and "\n" in self.tok.chars else None
        with self.cond:
            r = Request(self.next_id, ids, max_new, temperature, newline, max(lines, 1))
            self.engine.submit(r)
            self.next_id += 1
            event = self.done[r.id] = threading.Event()
            self.cond.notify()
        event.wait()
        return r

    def _loop(self) -> None:
        while True:
            with self.cond:
                while self.running and self.engine.idle:
                    self.cond.wait()
                if not self.running:
                    return
                for r in self.engine.step():
                    self.served += 1
                    self.done.pop(r.id).set()

    def stop(self) -> None:
        with self.cond:
            self.running = False
            self.cond.notify()
        self.thread.join()


def handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            c = worker.cfg
            self._send(
                200,
                {
                    "ok": True,
                    "served": worker.served,
                    "model": {"ctx": c.ctx, "dim": c.dim, "layers": c.layers},
                },
            )

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.close_connection = True
                return self._send(413, {"error": f"body over {MAX_BODY} bytes"})
            try:
                body = json.loads(self.rfile.read(length))
                temperature = body.get("temperature")
                start = time.perf_counter()
                r = worker.submit(
                    str(body["prompt"]),
                    int(body.get("max_new", 32)),
                    None if temperature is None else float(temperature),
                    int(body.get("lines", 0)),
                )
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e) or type(e).__name__})
            self._send(
                200,
                {
                    "text": worker.tok.decode(r.out),
                    "tokens": len(r.out),
                    "seconds": round(time.perf_counter() - start, 4),
                },
            )

        def log_message(self, *args):  # quiet by default
            pass

    return Handler


def make_server(params, cfg, tok, host="127.0.0.1", port=8000, **kw):
    worker = Worker(params, cfg, tok, **kw)
    httpd = ThreadingHTTPServer((host, port), handler(worker))
    httpd.worker = worker
    return httpd
