"""
Provenance and rerun checks for saved Story 1 runs: R-19, R-20 and R-21.

R-19 also runs on the real S8 window. That test needs the Volatility 75 dataset folder
and is skipped when the folder is absent. Every other test writes its own tiny fixture
under tmp_path and never reads the real files.
"""

import hashlib
import itertools
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vectorbt_research.reports import read_run
from vectorbt_research.runner import run_story_1_window

DATASET_DIR = Path(
    "/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped"
)
# The dataset folder holds one yearly file per year, from 2016 to 2026.
DATASET_YEARS = tuple(range(2016, 2027))
UTC_LABEL = "UTC, same exporter, not directly proven"
STARTING_CASH_CENTS = 1_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _ns(year, month, day, hour, minute):
    """
    Return a UTC instant as integer nanoseconds.
    """
    moment = datetime(year, month, day, hour, minute, tzinfo=UTC)
    return (moment - _EPOCH) // timedelta(microseconds=1) * 1_000


# S8: 2026-08-18 03:20-04:20 UTC, the one-hour history window.
S8_START_NS = _ns(2026, 8, 18, 3, 20)
S8_END_NS = _ns(2026, 8, 18, 4, 20)
# S7: the five hand-written test trades, in list order. Their starts lie inside S8.
S7_SIGNALS = [
    {
        "signal_id": "signal-01",
        "direction": "CALL",
        "start_time": "2026-08-18T03:22:00Z",
    },
    {
        "signal_id": "signal-02",
        "direction": "PUT",
        "start_time": "2026-08-18T03:21:00Z",
    },
    {
        "signal_id": "signal-03",
        "direction": "PUT",
        "start_time": "2026-08-18T03:21:00Z",
    },
    {
        "signal_id": "signal-04",
        "direction": "CALL",
        "start_time": "2026-08-18T03:24:00Z",
    },
    {
        "signal_id": "signal-05",
        "direction": "CALL",
        "start_time": "2026-08-18T03:23:00Z",
    },
]
# Four quotes on one UTC day. A signal that starts at 03:20:30 enters at the 03:20:40
# quote and expires at 03:21:30, so its exit is the 03:21:29 quote.
SMALL_QUOTES = [
    ("03:20:02.000", "99.00", "99.00"),
    ("03:20:40.000", "100.00", "100.00"),
    ("03:21:29.000", "101.00", "101.00"),
    ("03:21:31.000", "101.50", "101.50"),
]


def _write_yearly_fixture(folder, day, quotes):
    """
    Write a tiny yearly Parquet file and exporter CSV for one UTC day, and return folder.

    Each quote is (clock, bid, ask) with exact price text, or None when a price is empty.
    The Parquet file holds doubles, as the real export does, and the CSV holds the text.
    """
    folder.mkdir(parents=True)
    date_text = day.strftime("%Y.%m.%d")
    pq.write_table(
        pa.table(
            {
                "<DATE>": [date_text] * len(quotes),
                "<TIME>": [clock for clock, _, _ in quotes],
                "<BID>": pa.array(
                    [None if bid is None else float(bid) for _, bid, _ in quotes],
                    type=pa.float64(),
                ),
                "<ASK>": pa.array(
                    [None if ask is None else float(ask) for _, _, ask in quotes],
                    type=pa.float64(),
                ),
            }
        ),
        folder / f"Volatility75Index_{day.year}.parquet",
    )
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>"]
    for clock, bid, ask in quotes:
        lines.append(f"{date_text}\t{clock}\t{bid or ''}\t{ask or ''}")
    (folder / f"Volatility75Index_{day.year}.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return folder


def _two_window_runs(output, **inputs):
    """
    Save the same window run twice, each in a new folder, and return both read back.
    """
    first_dir = run_story_1_window(output_dir=output / "first", **inputs)
    second_dir = run_story_1_window(output_dir=output / "second", **inputs)
    assert first_dir != second_dir
    return read_run(first_dir), read_run(second_dir)


def _assert_same_report(first, second):
    """
    Assert that two saved runs match in manifest, summary and all three tables.

    The manifest holds no run ID and no wall time, so nothing is left out of the check.
    """
    assert first["manifest"] == second["manifest"]
    assert first["summary"] == second["summary"]
    for name in ("signals", "contracts", "cash"):
        pd.testing.assert_frame_equal(first[name], second[name])


def _assert_cash_reconciles(run, starting_cents):
    """
    Check the cash log against the contracts table, one contract at a time, in cents.

    Every cash row starts where the row before it ended. A contract that opened has an
    entry row that debits its stake. A settled contract also has a settlement row that
    adds its gross credit and releases its stake. An unresolved contract keeps its stake
    locked, with no settlement row.
    """
    cash = run["cash"].to_dict("records")
    contracts = run["contracts"].to_dict("records")
    for column in (
        "before_available_cents",
        "before_locked_cents",
        "available_cents",
        "locked_cents",
    ):
        assert run["cash"][column].dtype == "int64"
    assert cash[0]["event"] == "initial"
    assert cash[0]["available_cents"] == starting_cents
    assert cash[0]["locked_cents"] == 0
    for before, after in itertools.pairwise(cash):
        assert after["before_available_cents"] == before["available_cents"]
        assert after["before_locked_cents"] == before["locked_cents"]
    for contract in contracts:
        status = contract["status"]
        assert status in ("settled", "skipped", "unresolved"), contract["signal_id"]
        rows = [row for row in cash if row["signal_id"] == contract["signal_id"]]
        if status == "skipped":
            assert rows == []
            continue
        stake = contract["stake_cents"]
        entry = rows[0]
        assert entry["event"] == "entry"
        assert entry["before_available_cents"] - entry["available_cents"] == stake
        assert entry["locked_cents"] - entry["before_locked_cents"] == stake
        if status == "settled":
            assert len(rows) == 2
            settlement = rows[1]
            assert settlement["event"] == "settlement"
            gained = (
                settlement["available_cents"] - settlement["before_available_cents"]
            )
            assert gained == contract["gross_credit_cents"]
            released = settlement["before_locked_cents"] - settlement["locked_cents"]
            assert released == stake
            assert contract["net_cents"] == contract["gross_credit_cents"] - stake
        else:
            assert len(rows) == 1
    settled_net = sum(c["net_cents"] for c in contracts if c["status"] == "settled")
    open_stakes = sum(
        c["stake_cents"] for c in contracts if c["status"] == "unresolved"
    )
    assert cash[-1]["available_cents"] == starting_cents + settled_net - open_stakes
    assert cash[-1]["locked_cents"] == open_stakes


def _sha256(path):
    """
    Return the SHA-256 of a file, read in 1 MiB chunks.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rerun_on_tiny_fixture_gives_identical_semantic_reports(tmp_path):
    """
    R-19: a rerun of frozen inputs gives identical semantic reports (tiny fixture).

    Two frozen signals settle on one tiny day. The manifest, summary and tables must
    match between the two runs, and the integer cash must reconcile contract by contract.
    """
    dataset = _write_yearly_fixture(
        tmp_path / "dataset", date(2026, 8, 18), SMALL_QUOTES
    )
    first, second = _two_window_runs(
        tmp_path / "runs",
        dataset_folder=dataset,
        window_start_ns=_ns(2026, 8, 18, 3, 20),
        window_end_ns=_ns(2026, 8, 18, 3, 21),
        frozen_signals=[
            {
                "signal_id": "signal-call",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:30Z",
            },
            {
                "signal_id": "signal-put",
                "direction": "PUT",
                "start_time": "2026-08-18T03:20:30Z",
            },
        ],
        scenario="bid",
    )
    _assert_same_report(first, second)
    _assert_cash_reconciles(first, STARTING_CASH_CENTS)
    contracts = first["contracts"].set_index("signal_id")
    assert contracts.loc["signal-call", "stake_cents"] == 500
    assert contracts.loc["signal-call", "gross_credit_cents"] == 976
    assert contracts.loc["signal-put", "stake_cents"] == 499
    assert contracts.loc["signal-put", "gross_credit_cents"] == 0
    assert first["cash"]["available_cents"].iloc[-1] == 999977
    assert first["cash"]["locked_cents"].iloc[-1] == 0


@pytest.mark.skipif(
    not DATASET_DIR.is_dir(), reason="Volatility75 dataset folder is absent"
)
def test_rerun_on_real_s8_window_gives_identical_semantic_reports(tmp_path):
    """
    R-19: a rerun of frozen inputs gives identical semantic reports (real S8 window).

    The S8 window and the five S7 trades run twice. The source files must have the same
    SHA-256 before and after, because the run only reads them.
    """
    sources = [
        DATASET_DIR / "Volatility75Index_2026.parquet",
        DATASET_DIR / "Volatility75Index_2026.csv",
    ]
    hashes_before = [_sha256(path) for path in sources]
    first, second = _two_window_runs(
        tmp_path / "runs",
        dataset_folder=DATASET_DIR,
        window_start_ns=S8_START_NS,
        window_end_ns=S8_END_NS,
        frozen_signals=S7_SIGNALS,
        scenario="bid",
    )
    assert [_sha256(path) for path in sources] == hashes_before
    _assert_same_report(first, second)
    _assert_cash_reconciles(first, STARTING_CASH_CENTS)
    assert first["manifest"]["counts"]["signals"] == 5


@pytest.mark.parametrize("year", DATASET_YEARS)
def test_window_before_cutoff_is_labelled_utc_not_directly_proven(tmp_path, year):
    """
    R-20: a dataset window before 2025-10-27 is labelled exactly UTC, same exporter, not
    directly proven.

    One tiny fixture per dataset year, all UTC. Only the two owner-confirmed sample dates
    are directly proven, so every fixture day here carries the exact label in the saved
    manifest. The locked plain summary holds only its seven fields, so it carries no
    time-zone line and must not claim any proof.
    """
    day = date(year, 1, 5)
    dataset = _write_yearly_fixture(tmp_path / "dataset", day, SMALL_QUOTES)
    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_ns(year, 1, 5, 3, 20),
        window_end_ns=_ns(year, 1, 5, 3, 21),
        frozen_signals=[
            {
                "signal_id": "signal-call",
                "direction": "CALL",
                "start_time": f"{day.isoformat()}T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )
    run = read_run(run_dir)
    manifest = run["manifest"]
    assert manifest["windows"][0]["utc_status"] == UTC_LABEL
    assert manifest["windows"][0]["utc_offset_seconds"] == 0
    assert "owner-confirmed" not in manifest["utc_proof_references"]
    assert "proven" not in run["summary"].casefold()


@pytest.mark.parametrize(
    "day", [date(2025, 10, 27), date(2026, 8, 18)], ids=["2025-10-27", "2026-08-18"]
)
def test_sample_dates_are_directly_proven_utc_with_zero_offset(tmp_path, day):
    """
    R-21: 2025-10-27 and 2026-08-18 are directly proven UTC with no daylight-saving offset.

    The saved manifest marks each owner-confirmed sample date as directly proven with
    offset 0, and keeps its D8 proof reference.
    """
    dataset = _write_yearly_fixture(tmp_path / "dataset", day, SMALL_QUOTES)
    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_ns(day.year, day.month, day.day, 3, 20),
        window_end_ns=_ns(day.year, day.month, day.day, 3, 21),
        frozen_signals=[
            {
                "signal_id": "signal-call",
                "direction": "CALL",
                "start_time": f"{day.isoformat()}T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )
    manifest = read_run(run_dir)["manifest"]
    assert manifest["windows"][0]["utc_status"] == "directly proven"
    assert manifest["windows"][0]["utc_offset_seconds"] == 0
    assert day.isoformat() in manifest["utc_proof_references"]
    assert "D8" in manifest["utc_proof_references"]
