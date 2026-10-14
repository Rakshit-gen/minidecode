import re

from minidecode.corpus import generate, problem, valid_rate

LINE = re.compile(r"\d\d:\d\d:\d\d (GET|POST|PUT|DELETE) /\S* \d{3} \d+ms")


def test_lines_have_the_log_shape():
    text = generate(500, seed=1)
    lines = text.splitlines()
    assert len(lines) == 500 and text.endswith("\n")
    assert all(LINE.fullmatch(x) for x in lines)


def test_same_seed_same_text_and_clock_moves_forward():
    assert generate(50, seed=3) == generate(50, seed=3) != generate(50, seed=4)
    times = [x[:8] for x in generate(100, seed=5, block=100).splitlines()]
    assert times == sorted(times)  # this seed starts well clear of midnight


def test_blocks_spread_over_the_day():
    hours = {x[:2] for x in generate(10000, seed=6).splitlines()}
    assert len(hours) >= 18


def test_every_generated_line_passes_and_mistakes_are_named():
    assert valid_rate(generate(2000, seed=9)) == 1.0
    assert problem("12:00:00 DELETE /login 204 3ms") == "DELETE not used on /login"
    assert problem("12:00:00 POST /login 200 3ms") == "status 200 after POST"
    assert problem("25:00:00 GET / 200 3ms") == "bad time"
    assert problem("12:00:00 GET /api/orders/0 200 3ms") == "unknown path"
    assert problem("12:00:00 GET / 200") == "not a log line"
    assert valid_rate("12:00:00 GET / 200 3ms\nnope\npartial") == 0.5
