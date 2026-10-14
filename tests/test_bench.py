from minidecode.bench import Timing, table, timed


def test_table_and_timing():
    calls = []
    t = timed("x", 10, lambda: calls.append(1), repeat=2)
    assert len(calls) == 3 and t.seconds >= 0  # one warm-up run plus two timed
    out = table([Timing("slow", 100, 2.0), Timing("fast", 100, 0.5)])
    assert "slow" in out and out.splitlines()[2].endswith("4.00x")
