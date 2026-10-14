import json
import threading
import urllib.error
import urllib.request

import pytest

from minidecode.generate import generate_cached
from minidecode.model import Config, init
from minidecode.server import make_server
from minidecode.tokenizer import CharTokenizer

TOK = CharTokenizer.fit("0123456789:\n abc")
CFG = Config(vocab=TOK.size, ctx=32, dim=16, layers=1, heads=2)
P = init(CFG, seed=0)


@pytest.fixture
def url():
    httpd = make_server(P, CFG, TOK, port=0, slots=3)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.worker.stop()


def post(url, body):
    req = urllib.request.Request(url + "/generate", json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_concurrent_requests_match_single_decoding(url):
    prompts = ["12:", "3", "ab", "0:1", "c", "9"]
    results = [None] * len(prompts)

    def go(i):
        results[i] = post(url, {"prompt": prompts[i], "max_new": 10, "temperature": 0})

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p, (code, body) in zip(prompts, results, strict=True):
        assert code == 200
        want = generate_cached(P, CFG, list(TOK.encode(p)), 10, temperature=0)
        assert body["text"] == TOK.decode(want)
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        assert json.loads(r.read())["served"] == len(prompts)


def test_bad_requests_get_400(url):
    assert post(url, {"prompt": "zz", "max_new": 3})[0] == 400  # not in the vocabulary
    assert post(url, {"prompt": "1", "max_new": 99})[0] == 400  # past the context
    assert post(url, {"max_new": 3})[0] == 400


def test_huge_body_is_refused_without_reading_it(url):
    req = urllib.request.Request(url + "/generate", b"{}", method="POST")
    req.add_header("Content-Length", str(10**9))
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 413
    req.headers["Content-length"] = "-1"
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400 and b"Content-Length" in e.value.read()
