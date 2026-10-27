import json
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vectorbt_research.reports import read_run, write_run

# Times are epoch nanoseconds. 2026-08-18 12:00:00 UTC is epoch second 1787054400.
# Money is integer cents. Prices are integers at price_scale 100.
# Hash literals are placeholders for this serialization fixture, not verified source hashes.
# Hash literals are placeholders for this serialization fixture, not verified source hashes.
_FILES = (
    "manifest.json",
    "signals.parquet",
    "contracts.parquet",
    "cash.parquet",
    "summary.txt",
)

_MANIFEST = {
    "schema_version": 1,
    "label": "MT5 simulation",
    "engine": "vectorbt_research",
    "purpose": "story 1 report contract fixture",
    "code_hash": "a" * 64,
    "rule_hash": "b" * 64,
    "lock_hash": "c" * 64,
    "strategy": {
        "source": "not applicable",
        "hash": "not applicable",
        "parameters": "not applicable",
        "seed": "not applicable",
    },
    "scenario": "bid",
    "windows": [
        {
            "start_timestamp_ns": 1_787_054_400_000_000_000,
            "end_timestamp_ns": 1_787_058_000_000_000_000,
            "utc_status": "directly proven",
            "utc_offset_seconds": 0,
        }
    ],
    "utc_proof_references": "DR-2 D8: owner-confirmed UTC sample date 2026-08-18; synthetic quote fixture",
    "holdout": "not applicable",
    "sources": [
        {
            "path": "in-memory fixture",
            "sha256": "d" * 64,
            "first_row": 1,
            "last_row": 2,
            "price_scale": 100,
        }
    ],
    "quote_provenance": "frozen in-memory fixture",
    "policies": {
        "symbol": "R_75",
        "directions": {"CALL": "Rise", "PUT": "Fall"},
        "expiry": "start plus 60 seconds",
        "net_payout_basis_points": 9520,
        "tie_rule": "equal exit price loses",
        "rounding": "cent floor",
        "minimum_stake_cents": 35,
        "stake_basis_points_of_available_cash": 5,
        "cash_rule": "available cash excludes locked stakes",
        "max_open_contracts": 5,
        "stale_limit_seconds": 10,
        "missing_side_rule": "row skipped and counted",
        "order": "timestamp, then source ordinal, then signal order",
    },
    "information_availability": "signal visible from its availability time only",
    "selected_prices": (
        "entry first tick at or after start; exit last tick at or before expiry"
    ),
    "quote_diagnostics": {"skipped_rows": 0, "missing_bid": 0, "missing_ask": 0},
    "counts": {
        "signals": 1,
        "contracts": 1,
        "skipped_signals": 0,
        "unresolved_contracts": 0,
    },
    "completeness": {"status": "complete"},
    "gate": {"status": "not evaluated", "evidence": "not applicable"},
}

_SIGNAL_ROWS = [
    {
        "signal_id": "signal-1",
        "direction": "CALL",
        "requested_start_timestamp_ns": 1_787_054_400_000_000_000,
        "available_timestamp_ns": 1_787_054_400_000_000_000,
    }
]

_CONTRACT_ROWS = [
    {
        "signal_id": "signal-1",
        "direction": "CALL",
        "price_reference": "bid",
        "status": "settled",
        "reason": "settled at expiry",
        "outcome": "win",
        "entry_ordinal": 1,
        "exit_ordinal": 2,
        "entry_timestamp_ns": 1_787_054_402_000_000_000,
        "expiry_timestamp_ns": 1_787_054_460_000_000_000,
        "exit_timestamp_ns": 1_787_054_460_000_000_000,
        "entry_spot": 10_000,
        "exit_spot": 10_100,
        "stake_cents": 500,
        "gross_credit_cents": 976,
        "net_cents": 476,
    }
]

_CASH_ROWS = [
    {
        "event": "initial",
        "signal_id": "",
        "before_available_cents": 0,
        "before_locked_cents": 0,
        "available_cents": 1_000_000,
        "locked_cents": 0,
    },
    {
        "event": "entry",
        "signal_id": "signal-1",
        "before_available_cents": 1_000_000,
        "before_locked_cents": 0,
        "available_cents": 999_500,
        "locked_cents": 500,
    },
    {
        "event": "settlement",
        "signal_id": "signal-1",
        "before_available_cents": 999_500,
        "before_locked_cents": 500,
        "available_cents": 1_000_476,
        "locked_cents": 0,
    },
]

_RECORD = {
    "manifest": _MANIFEST,
    "signals": _SIGNAL_ROWS,
    "contracts": _CONTRACT_ROWS,
    "cash": _CASH_ROWS,
}

_INTEGER_COLUMNS = {
    "signals.parquet": ("requested_start_timestamp_ns", "available_timestamp_ns"),
    "contracts.parquet": (
        "entry_ordinal",
        "exit_ordinal",
        "entry_timestamp_ns",
        "expiry_timestamp_ns",
        "exit_timestamp_ns",
        "entry_spot",
        "exit_spot",
        "stake_cents",
        "gross_credit_cents",
        "net_cents",
    ),
    "cash.parquet": (
        "before_available_cents",
        "before_locked_cents",
        "available_cents",
        "locked_cents",
    ),
}

# The summary is one "Field: value" line per field. Other nonempty lines are rejected.
_SUMMARY_FIELDS = {
    "Label": "MT5 simulation",
    "Scenario": "bid",
    "Final available": "1,000,476 cents (10,004.76 USD)",
    "Locked": "0 cents",
    "Skipped signals": "0",
    "Unresolved contracts": "0",
    "Status": "complete",
}


def _summary_fields(summary):
    # Match the locked E2E parser: seven field words, exactly once each.
    for word in (
        "label",
        "scenario",
        "available",
        "locked",
        "skipped",
        "unresolved",
        "status",
    ):
        assert len(re.findall(rf"\b{word}\b", summary.casefold())) == 1, word
    fields = {}
    for line in summary.splitlines():
        if line.strip():
            name, separator, value = line.partition(": ")
            assert separator and name not in fields, line
            fields[name] = value
    return fields


@pytest.mark.parametrize("scenario", ["bid", "mid", "ask"])
def test_saved_run_writes_and_reopens_all_artifacts(tmp_path, scenario):
    """a saved run has manifest, signals, contracts, cash and summary per scenario, labelled MT5 simulation"""
    record = deepcopy(_RECORD)
    record["manifest"]["scenario"] = scenario
    record["contracts"][0]["price_reference"] = scenario
    summary_fields = {**_SUMMARY_FIELDS, "Scenario": scenario}
    run_dir = write_run(record, tmp_path)
    assert isinstance(run_dir, Path)
    assert run_dir.parent == tmp_path
    # Each saved run gets its own unique folder.
    second_dir = write_run(record, tmp_path)
    assert isinstance(second_dir, Path)
    assert second_dir.parent == tmp_path
    assert second_dir != run_dir
    for name in _FILES:
        assert (run_dir / name).is_file(), name
    for name, columns in _INTEGER_COLUMNS.items():
        schema = pq.read_schema(run_dir / name)
        for column in columns:
            assert pa.types.is_integer(schema.field(column).type), column

    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert json.dumps(saved_manifest, sort_keys=True) == json.dumps(
        record["manifest"], sort_keys=True
    )
    for name in ("signals", "contracts", "cash"):
        saved = pq.read_table(run_dir / f"{name}.parquet")
        assert set(saved.column_names) == set(record[name][0])
        assert saved.to_pylist() == record[name]
    saved_summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert _summary_fields(saved_summary) == summary_fields

    run = read_run(run_dir)
    assert set(run) == {"manifest", "signals", "contracts", "cash", "summary"}
    for name in ("signals", "contracts", "cash"):
        assert isinstance(run[name], pd.DataFrame)
        for column in _INTEGER_COLUMNS[f"{name}.parquet"]:
            assert pd.api.types.is_integer_dtype(run[name][column].dtype), column
            assert not run[name][column].isna().any(), column
        for column in _INTEGER_COLUMNS[f"{name}.parquet"]:
            assert pd.api.types.is_integer_dtype(run[name][column].dtype), column
            assert not run[name][column].isna().any(), column
    assert run["summary"] == saved_summary
    assert run["manifest"]["label"] == "MT5 simulation"
    # Canonical JSON, so an int, float or bool with the same value still fails.
    manifest = json.dumps(run["manifest"], sort_keys=True)
    assert manifest == json.dumps(record["manifest"], sort_keys=True)
    assert run["signals"].to_dict("records") == _SIGNAL_ROWS
    assert run["contracts"].to_dict("records") == record["contracts"]
    assert run["cash"].to_dict("records") == _CASH_ROWS
    assert _summary_fields(run["summary"]) == summary_fields
