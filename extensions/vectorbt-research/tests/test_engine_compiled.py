"""Randomized parity checks for the compiled and Python ledger loops."""

import random

from vectorbt_research.engine import run_ledger, run_ledger_chunks

_T0 = 1_787_023_200_000_000_000
_SEC = 1_000_000_000
_SCENARIOS = ("bid", "mid", "ask")


def _schedule(rng, number):
    count = rng.randint(0, 45)
    tick_seconds = sorted(rng.randint(-5, 180) for _ in range(count))
    ticks = []
    for ordinal, second in enumerate(tick_seconds, start=1):
        base = rng.randint(19_000, 21_000)
        ticks.append(
            (
                _T0 + second * _SEC,
                ordinal,
                base - rng.randint(0, 4),
                base,
                base + rng.randint(0, 4),
            )
        )
    signals = [
        {
            "signal_id": f"{number}-{index}",
            "direction": rng.choice(("CALL", "PUT")),
            "start_time_ns": _T0 + rng.randint(-5, 170) * _SEC,
            "availability_ordinal": rng.randint(1, count + 3),
        }
        for index in range(rng.randint(0, 12))
    ]
    cash = rng.randint(0, 250_000)
    window_end = None
    if rng.choice((False, True)):
        window_end = _T0 + rng.randint(10, 130) * _SEC
    return ticks, signals, cash, window_end


def _chunks_at_random_splits(rng, ticks):
    if not ticks:
        return [[], []]
    cuts = sorted(
        rng.sample(range(len(ticks) + 1), k=rng.randint(1, min(5, len(ticks) + 1)))
    )
    chunks = []
    previous = 0
    for cut in cuts:
        chunks.append(ticks[previous:cut])
        previous = cut
    chunks.append(ticks[previous:])
    return chunks


def test_compiled_ledger_matches_python_for_random_schedules_and_splits():
    rng = random.Random(20261027)
    for number in range(250):
        ticks, signals, cash, window_end = _schedule(rng, number)
        chunks = _chunks_at_random_splits(rng, ticks)
        for scenario in _SCENARIOS:
            options = {
                "scenario": scenario,
                "starting_cash_cents": cash,
                "window_end_ns": window_end,
            }
            expected = run_ledger(ticks, signals, **options)
            actual = run_ledger(ticks, signals, compiled=True, **options)
            assert actual == expected, f"schedule={number}, scenario={scenario}, whole"
            expected_chunks = run_ledger_chunks(chunks, signals, **options)
            actual_chunks = run_ledger_chunks(chunks, signals, compiled=True, **options)
            assert actual_chunks == expected_chunks, (
                f"schedule={number}, scenario={scenario}, chunks={chunks}"
            )
