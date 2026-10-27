from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from vectorbt_research.runner import run_story_1

_BASE = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)

_REQUIRED_COLUMNS = {
    "signals": {"signal_id", "direction"},
    "contracts": {
        "signal_id",
        "direction",
        "status",
        "reason",
        "entry_ordinal",
        "exit_ordinal",
        "entry_timestamp_ns",
        "expiry_timestamp_ns",
        "exit_timestamp_ns",
        "stake_cents",
        "gross_credit_cents",
        "net_cents",
        "outcome",
    },
    "cash": {"event", "signal_id", "available_cents", "locked_cents"},
}
_MONEY_COLUMNS = {
    "contracts": ("stake_cents", "gross_credit_cents", "net_cents"),
    "cash": ("available_cents", "locked_cents"),
}


def _timestamp(seconds: int) -> str:
    return (
        (_BASE + timedelta(seconds=seconds))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _quote(ordinal: int, seconds: int, price: str = "100.00", *, bid="set", ask="set"):
    return {
        "timestamp": _timestamp(seconds),
        "ordinal": ordinal,
        "bid": price if bid == "set" else bid,
        "ask": price if ask == "set" else ask,
    }


def _signal(signal_id: str = "signal-1", direction: str = "CALL") -> dict[str, str]:
    return {"signal_id": signal_id, "direction": direction, "start_time": _timestamp(0)}


def _basic_quotes(entry: str = "100.00", exit: str = "101.00"):
    return [_quote(1, 2, entry), _quote(2, 60, exit)]


def _read_run(
    tmp_path,
    name,
    quote_rows,
    frozen_signals,
    *,
    scenario="bid",
    starting_cash_cents=1_000_000,
):
    run_dir = Path(
        run_story_1(
            quote_rows=quote_rows,
            frozen_signals=frozen_signals,
            scenario=scenario,
            output_dir=tmp_path / name,
            starting_cash_cents=starting_cash_cents,
        )
    )
    assert run_dir.is_dir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    signals = pd.read_parquet(run_dir / "signals.parquet", engine="pyarrow")
    contracts = pd.read_parquet(run_dir / "contracts.parquet", engine="pyarrow")
    cash = pd.read_parquet(run_dir / "cash.parquet", engine="pyarrow")
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert manifest["schema_version"]
    assert manifest["label"] == "MT5 simulation"
    assert manifest["scenario"] == scenario
    tables = {"signals": signals, "contracts": contracts, "cash": cash}
    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        assert required_columns <= set(tables[table_name].columns)
    for table_name, columns in _MONEY_COLUMNS.items():
        for column in columns:
            assert pd.api.types.is_integer_dtype(tables[table_name][column].dtype)
            assert not tables[table_name][column].isna().any()
    return {
        "dir": run_dir,
        "manifest": manifest,
        "signals": signals,
        "contracts": contracts,
        "cash": cash,
        "summary": summary,
    }


def _contract(report, signal_id="signal-1"):
    matches = report["contracts"].loc[report["contracts"]["signal_id"] == signal_id]
    assert len(matches) == 1
    return matches.iloc[0]


def _state(row):
    return row["available_cents"], row["locked_cents"]


# The plain summary holds only one "Field: value" line per required field (blank lines
# allowed, any order). Each field word appears exactly once, so nothing can contradict it.
_SUMMARY_FIELDS = (
    "Label",
    "Scenario",
    "Final available",
    "Locked",
    "Skipped signals",
    "Unresolved contracts",
    "Status",
)
_SUMMARY_WORDS = (
    "label",
    "scenario",
    "available",
    "locked",
    "skipped",
    "unresolved",
    "status",
)


def _summary_fields(summary):
    lowered = summary.casefold()
    for word in _SUMMARY_WORDS:
        assert len(re.findall(rf"\b{word}\b", lowered)) == 1, word
    fields = {}
    for line in summary.splitlines():
        name, separator, value = line.partition(": ")
        assert not line.strip() or (separator and name in _SUMMARY_FIELDS), line
        if separator and name in _SUMMARY_FIELDS:
            fields[name] = value
    return fields


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "bid",
            id=(
                "Given 10,000.00 USD free, when a 5.00 Rise trade is placed (0.05% "
                "of free cash), then 5.00 is locked while open; when it wins, the "
                "gross credit is 9.76 and free cash ends at 10,004.76."
            ),
        )
    ],
)
def test_e2e_1_money_win_loss_and_tie_are_persisted(tmp_path, scenario):
    win = _read_run(tmp_path, "win", _basic_quotes(), [_signal()], scenario=scenario)
    contract = _contract(win)
    assert contract["direction"] == "CALL"
    assert contract["stake_cents"] == 500
    assert contract["gross_credit_cents"] == 976
    assert contract["net_cents"] == 476
    assert contract["outcome"] == "win"
    assert contract["status"] == "settled"
    assert _state(win["cash"].loc[win["cash"]["event"] == "entry"].iloc[0]) == (
        999500,
        500,
    )
    assert _state(win["cash"].loc[win["cash"]["event"] == "settlement"].iloc[-1]) == (
        1000476,
        0,
    )

    for case, exit_price in (("loss", "99.00"), ("tie", "100.00")):
        report = _read_run(
            tmp_path,
            case,
            _basic_quotes(exit=exit_price),
            [_signal()],
            scenario=scenario,
        )
        contract = _contract(report)
        assert contract["gross_credit_cents"] == 0
        assert contract["net_cents"] == -500
        assert contract["outcome"] == "loss"
        assert contract["status"] == "settled"
        settled = report["cash"].loc[report["cash"]["event"] == "settlement"]
        assert _state(settled.iloc[-1]) == (999500, 0)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "bid", id=("Given 9,995.00 free, the stake is 4.99 (rounded down to cents)")
        )
    ],
)
def test_e2e_2_rounded_stake_and_gross_credit_are_persisted(tmp_path, scenario):
    report = _read_run(
        tmp_path,
        "round-down",
        _basic_quotes(),
        [_signal()],
        scenario=scenario,
        starting_cash_cents=999500,
    )
    contract = _contract(report)
    assert contract["stake_cents"] == 499
    assert contract["gross_credit_cents"] == 974
    assert contract["net_cents"] == 475
    assert contract["status"] == "settled"
    assert contract["outcome"] == "win"
    assert _state(report["cash"].loc[report["cash"]["event"] == "entry"].iloc[0]) == (
        999001,
        499,
    )
    settled = report["cash"].loc[report["cash"]["event"] == "settlement"]
    assert _state(settled.iloc[-1]) == (999975, 0)


@pytest.mark.parametrize(
    "scenario",
    [pytest.param("bid", id=("given 699.99 free, the stake is 0.35 (the minimum)"))],
)
def test_e2e_3_minimum_stake_and_gross_credit_are_persisted(tmp_path, scenario):
    report = _read_run(
        tmp_path,
        "minimum",
        _basic_quotes(),
        [_signal()],
        scenario=scenario,
        starting_cash_cents=69999,
    )
    contract = _contract(report)
    assert contract["stake_cents"] == 35
    assert contract["gross_credit_cents"] == 68
    assert contract["net_cents"] == 33
    assert contract["status"] == "settled"
    assert contract["outcome"] == "win"
    assert _state(report["cash"].loc[report["cash"]["event"] == "entry"].iloc[0]) == (
        69964,
        35,
    )
    settled = report["cash"].loc[report["cash"]["event"] == "settlement"]
    assert _state(settled.iloc[-1]) == (70032, 0)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "bid", id=("given 0.34 free, no trade is placed and the reason is logged")
        )
    ],
)
def test_e2e_4_insufficient_cash_is_persisted_without_a_debit(tmp_path, scenario):
    report = _read_run(
        tmp_path,
        "insufficient-cash",
        _basic_quotes(),
        [_signal()],
        scenario=scenario,
        starting_cash_cents=34,
    )
    contract = _contract(report)
    assert contract["status"] == "skipped"
    reason = str(contract["reason"]).lower()
    assert "insufficient" in reason and "cash" in reason
    assert not (report["cash"]["event"] == "entry").any()
    states = {
        _state(row)
        for _, row in report["cash"][["available_cents", "locked_cents"]].iterrows()
    }
    assert states == {(34, 0)}


@pytest.mark.parametrize(
    "scenarios",
    [
        pytest.param(
            ("bid", "mid", "ask"),
            id=(
                "Given a start at 12:00:00 PM UTC, when a trade is placed, then "
                "entry is the first price at or after the start (12:00:02 PM), "
                "expiry is 12:01:00 PM, and exit is the last price at or before "
                "expiry; an expiry price 10 s old or less settles, one older than "
                "10 s stays unresolved, locked and flagged; a first price after "
                "expiry means no trade; an equal exit price loses; rows with an "
                "empty bid or ask are skipped and counted."
            ),
        )
    ],
)
def test_e2e_5_tick_boundaries_missing_quotes_and_tie_are_persisted(
    tmp_path, scenarios
):
    quotes = [
        _quote(20, 2),
        _quote(21, 2, "99.00"),
        _quote(22, 58, "100.50"),
        _quote(23, 60, "101.00"),
        _quote(24, 60, "102.00"),
        _quote(25, 61, "999.00"),
    ]
    boundary = _read_run(
        tmp_path, "boundary", quotes, [_signal()], scenario=scenarios[0]
    )
    contract = _contract(boundary)
    assert contract["entry_ordinal"] == 20
    assert contract["exit_ordinal"] == 24
    assert contract["entry_timestamp_ns"] == pd.Timestamp(_timestamp(2)).value
    assert contract["expiry_timestamp_ns"] == pd.Timestamp(_timestamp(60)).value
    assert contract["exit_timestamp_ns"] == pd.Timestamp(_timestamp(60)).value
    assert contract["gross_credit_cents"] == 976
    assert contract["status"] == "settled"
    assert contract["outcome"] == "win"

    age_ten = _read_run(
        tmp_path,
        "age-10",
        [_quote(30, 2), _quote(31, 50, "101.00"), _quote(32, 61, "999.00")],
        [_signal()],
        scenario=scenarios[0],
    )
    contract = _contract(age_ten)
    assert contract["status"] == "settled"
    assert contract["exit_timestamp_ns"] == pd.Timestamp(_timestamp(50)).value
    assert contract["expiry_timestamp_ns"] == pd.Timestamp(_timestamp(60)).value
    assert contract["outcome"] == "win"
    settled = age_ten["cash"].loc[age_ten["cash"]["event"] == "settlement"]
    assert _state(settled.iloc[-1]) == (1000476, 0)

    age_eleven = _read_run(
        tmp_path,
        "age-11",
        [_quote(40, 2), _quote(41, 49, "102.00"), _quote(42, 61, "999.00")],
        [_signal()],
        scenario=scenarios[0],
    )
    contract = _contract(age_eleven)
    assert contract["status"] == "unresolved"
    assert str(contract["reason"]).strip()
    assert contract["outcome"] == ""
    assert contract["entry_ordinal"] == 40
    assert contract["entry_timestamp_ns"] == pd.Timestamp(_timestamp(2)).value
    assert contract["expiry_timestamp_ns"] == pd.Timestamp(_timestamp(60)).value
    assert contract["entry_spot"] == 20000
    assert contract["stake_cents"] == 500
    assert contract["exit_ordinal"] == 0
    assert contract["exit_timestamp_ns"] == 0
    assert contract["exit_spot"] == 0
    assert contract["gross_credit_cents"] == 0
    assert contract["net_cents"] == 0
    assert not (age_eleven["cash"]["event"] == "settlement").any()
    assert _state(age_eleven["cash"].iloc[-1]) == (999500, 500)

    late = _read_run(
        tmp_path,
        "late-entry",
        [_quote(50, 61, "101.00")],
        [_signal()],
        scenario=scenarios[0],
    )
    contract = _contract(late)
    assert contract["status"] == "skipped"
    reason = str(contract["reason"]).lower()
    assert "late" in reason or "expiry" in reason
    assert not (late["cash"]["event"] == "entry").any()
    assert _state(late["cash"].iloc[-1]) == (1_000_000, 0)

    tie = _read_run(
        tmp_path,
        "equal-exit",
        _basic_quotes(exit="100.00"),
        [_signal()],
        scenario=scenarios[0],
    )
    assert _contract(tie)["gross_credit_cents"] == 0
    assert _contract(tie)["outcome"] == "loss"
    assert _contract(tie)["status"] == "settled"
    assert _state(tie["cash"].loc[tie["cash"]["event"] == "settlement"].iloc[-1]) == (
        999500,
        0,
    )

    missing = [
        _quote(60, 2, bid=None),
        _quote(61, 2, ask=None),
        _quote(62, 2, bid=None, ask=None),
        _quote(63, 3),
        _quote(64, 60, "101.00"),
        _quote(65, 61, "999.00"),
    ]
    for scenario in scenarios:
        report = _read_run(
            tmp_path, f"missing-{scenario}", missing, [_signal()], scenario=scenario
        )
        assert report["manifest"]["quote_diagnostics"] == {
            "skipped_rows": 3,
            "missing_bid": 2,
            "missing_ask": 2,
        }
        contract = _contract(report)
        assert contract["entry_ordinal"] == 63
        assert contract["exit_ordinal"] == 64
        assert contract["status"] == "settled"
        assert contract["outcome"] == "win"


@pytest.mark.parametrize(
    "scenarios",
    [
        pytest.param(
            ("bid", "mid", "ask"),
            id=(
                "Given a fresh 10,000.00 USD run and five accepted trades opened in "
                "signal order before any settles, when a sixth signal arrives, then "
                "it is skipped and logged, free cash is 9,975.04 with 24.96 locked, "
                "and when all five win the balance ends at 10,023.76; the bid, mid "
                "and ask results are saved as three separate reports, each labelled "
                '"MT5 simulation". Each scenario starts independently at 10,000.00 '
                "USD, and locked cash returns to 0.00 after all five resolve."
            ),
        )
    ],
)
def test_e2e_6_five_open_limit_and_three_scenario_reports_are_persisted(
    tmp_path, scenarios
):
    ids = [f"signal-{index}" for index in range(1, 7)]
    signals = [_signal(signal_id) for signal_id in ids]
    run_dirs = set()
    for scenario in scenarios:
        report = _read_run(
            tmp_path,
            f"five-open-{scenario}",
            _basic_quotes(),
            signals,
            scenario=scenario,
        )
        run_dirs.add(report["dir"])
        assert report["signals"]["signal_id"].tolist() == ids
        contracts = report["contracts"].set_index("signal_id")
        stakes = [contracts.loc[sid, "stake_cents"] for sid in ids[:5]]
        assert stakes == [500, 499, 499, 499, 499]
        credits = [contracts.loc[sid, "gross_credit_cents"] for sid in ids[:5]]
        assert credits == [976, 974, 974, 974, 974]
        assert [contracts.loc[sid, "status"] for sid in ids[:5]] == ["settled"] * 5
        assert [contracts.loc[sid, "outcome"] for sid in ids[:5]] == ["win"] * 5
        sixth = contracts.loc[ids[5]]
        assert sixth["status"] == "skipped"
        reason = str(sixth["reason"]).lower()
        assert "capacity" in reason or "limit" in reason
        initial = report["cash"].loc[report["cash"]["event"] == "initial"]
        assert _state(initial.iloc[0]) == (1_000_000, 0)
        entries = report["cash"].loc[report["cash"]["event"] == "entry"]
        assert entries["signal_id"].tolist() == ids[:5]
        assert _state(entries.iloc[-1]) == (997504, 2496)
        settlements = report["cash"].loc[report["cash"]["event"] == "settlement"]
        assert entries.index[-1] < settlements.index[0]
        settled = report["cash"].loc[report["cash"]["event"] == "settlement"]
        assert _state(settled.iloc[-1]) == (1002376, 0)
        summary = report["summary"]
        assert _summary_fields(summary) == {
            "Label": "MT5 simulation",
            "Scenario": scenario,
            "Final available": "1,002,376 cents (10,023.76 USD)",
            "Locked": "0 cents",
            "Skipped signals": "1",
            "Unresolved contracts": "0",
            "Status": "complete",
        }
    assert len(run_dirs) == 3
