import json
import re
import uuid
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


# Required manifest fields and groups from the approved evidence contract (R-18).
_REQUIRED_MANIFEST_PATHS = [
    pytest.param(("schema_version",), id="schema_version"),
    pytest.param(("label",), id="label"),
    pytest.param(("engine",), id="engine"),
    pytest.param(("purpose",), id="purpose"),
    pytest.param(("code_hash",), id="code_hash"),
    pytest.param(("rule_hash",), id="rule_hash"),
    pytest.param(("lock_hash",), id="lock_hash"),
    pytest.param(("strategy",), id="strategy"),
    pytest.param(("strategy", "source"), id="strategy_source"),
    pytest.param(("strategy", "hash"), id="strategy_hash"),
    pytest.param(("strategy", "parameters"), id="strategy_parameters"),
    pytest.param(("strategy", "seed"), id="strategy_seed"),
    pytest.param(("scenario",), id="scenario"),
    pytest.param(("windows",), id="windows"),
    pytest.param(
        ("windows", 0, "start_timestamp_ns"), id="windows_0_start_timestamp_ns"
    ),
    pytest.param(("windows", 0, "end_timestamp_ns"), id="windows_0_end_timestamp_ns"),
    pytest.param(("windows", 0, "utc_status"), id="windows_0_utc_status"),
    pytest.param(
        ("windows", 0, "utc_offset_seconds"), id="windows_0_utc_offset_seconds"
    ),
    pytest.param(("utc_proof_references",), id="utc_proof_references"),
    pytest.param(("holdout",), id="holdout"),
    pytest.param(("sources",), id="sources"),
    pytest.param(("sources", 0, "path"), id="sources_0_path"),
    pytest.param(("sources", 0, "sha256"), id="sources_0_sha256"),
    pytest.param(("sources", 0, "first_row"), id="sources_0_first_row"),
    pytest.param(("sources", 0, "last_row"), id="sources_0_last_row"),
    pytest.param(("sources", 0, "price_scale"), id="sources_0_price_scale"),
    pytest.param(("quote_provenance",), id="quote_provenance"),
    pytest.param(("policies",), id="policies"),
    pytest.param(("policies", "symbol"), id="policies_symbol"),
    pytest.param(("policies", "directions"), id="policies_directions"),
    pytest.param(("policies", "directions", "CALL"), id="policies_directions_CALL"),
    pytest.param(("policies", "directions", "PUT"), id="policies_directions_PUT"),
    pytest.param(("policies", "expiry"), id="policies_expiry"),
    pytest.param(
        ("policies", "net_payout_basis_points"), id="policies_net_payout_basis_points"
    ),
    pytest.param(("policies", "tie_rule"), id="policies_tie_rule"),
    pytest.param(("policies", "rounding"), id="policies_rounding"),
    pytest.param(
        ("policies", "minimum_stake_cents"), id="policies_minimum_stake_cents"
    ),
    pytest.param(
        ("policies", "stake_basis_points_of_available_cash"),
        id="policies_stake_basis_points_of_available_cash",
    ),
    pytest.param(("policies", "cash_rule"), id="policies_cash_rule"),
    pytest.param(("policies", "max_open_contracts"), id="policies_max_open_contracts"),
    pytest.param(
        ("policies", "stale_limit_seconds"), id="policies_stale_limit_seconds"
    ),
    pytest.param(("policies", "missing_side_rule"), id="policies_missing_side_rule"),
    pytest.param(("policies", "order"), id="policies_order"),
    pytest.param(("information_availability",), id="information_availability"),
    pytest.param(("selected_prices",), id="selected_prices"),
    pytest.param(("quote_diagnostics",), id="quote_diagnostics"),
    pytest.param(
        ("quote_diagnostics", "skipped_rows"), id="quote_diagnostics_skipped_rows"
    ),
    pytest.param(
        ("quote_diagnostics", "missing_bid"), id="quote_diagnostics_missing_bid"
    ),
    pytest.param(
        ("quote_diagnostics", "missing_ask"), id="quote_diagnostics_missing_ask"
    ),
    pytest.param(("counts",), id="counts"),
    pytest.param(("counts", "signals"), id="counts_signals"),
    pytest.param(("counts", "contracts"), id="counts_contracts"),
    pytest.param(("counts", "skipped_signals"), id="counts_skipped_signals"),
    pytest.param(("counts", "unresolved_contracts"), id="counts_unresolved_contracts"),
    pytest.param(("completeness",), id="completeness"),
    pytest.param(("completeness", "status"), id="completeness_status"),
    pytest.param(("gate",), id="gate"),
    pytest.param(("gate", "status"), id="gate_status"),
    pytest.param(("gate", "evidence"), id="gate_evidence"),
]

# Every manifest integer must reject float and bool values.
_INTEGER_MANIFEST_PATHS = [
    pytest.param(("schema_version",), id="schema_version"),
    pytest.param(
        ("windows", 0, "start_timestamp_ns"), id="windows_0_start_timestamp_ns"
    ),
    pytest.param(("windows", 0, "end_timestamp_ns"), id="windows_0_end_timestamp_ns"),
    pytest.param(
        ("windows", 0, "utc_offset_seconds"), id="windows_0_utc_offset_seconds"
    ),
    pytest.param(("sources", 0, "first_row"), id="sources_0_first_row"),
    pytest.param(("sources", 0, "last_row"), id="sources_0_last_row"),
    pytest.param(("sources", 0, "price_scale"), id="sources_0_price_scale"),
    pytest.param(
        ("policies", "net_payout_basis_points"), id="policies_net_payout_basis_points"
    ),
    pytest.param(
        ("policies", "minimum_stake_cents"), id="policies_minimum_stake_cents"
    ),
    pytest.param(
        ("policies", "stake_basis_points_of_available_cash"),
        id="policies_stake_basis_points_of_available_cash",
    ),
    pytest.param(("policies", "max_open_contracts"), id="policies_max_open_contracts"),
    pytest.param(
        ("policies", "stale_limit_seconds"), id="policies_stale_limit_seconds"
    ),
    pytest.param(
        ("quote_diagnostics", "skipped_rows"), id="quote_diagnostics_skipped_rows"
    ),
    pytest.param(
        ("quote_diagnostics", "missing_bid"), id="quote_diagnostics_missing_bid"
    ),
    pytest.param(
        ("quote_diagnostics", "missing_ask"), id="quote_diagnostics_missing_ask"
    ),
    pytest.param(("counts", "signals"), id="counts_signals"),
    pytest.param(("counts", "contracts"), id="counts_contracts"),
    pytest.param(("counts", "skipped_signals"), id="counts_skipped_signals"),
    pytest.param(("counts", "unresolved_contracts"), id="counts_unresolved_contracts"),
]


def _parent_and_key(data, path):
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    return parent, path[-1]


def _save_manifest(run_dir, manifest):
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )


@pytest.mark.parametrize("path", _REQUIRED_MANIFEST_PATHS)
def test_run_missing_policy_or_provenance_fields_is_rejected(tmp_path, path):
    """a run missing policy or provenance fields is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    del parent[key]
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError):
        read_run(run_dir)


@pytest.mark.parametrize("path", _INTEGER_MANIFEST_PATHS)
@pytest.mark.parametrize("coerce", [float, bool], ids=["float", "bool"])
def test_run_with_non_integer_where_integer_is_required_is_rejected(
    tmp_path, path, coerce
):
    """a float or bool where an integer is required in the manifest is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    parent[key] = coerce(parent[key])
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError):
        read_run(run_dir)


@pytest.mark.parametrize("value", [1_000_000.9, 1_000_000.0, True])
def test_money_column_rejects_float_and_bool(tmp_path, value):
    """a float or bool in a money column makes write_run raise TypeError"""
    record = deepcopy(_RECORD)
    record["cash"][0]["available_cents"] = value

    with pytest.raises(TypeError, match="available_cents must be an integer"):
        write_run(record, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_leaves_no_run_folder(tmp_path, monkeypatch):
    """a write that fails part-way leaves no run-* folder behind"""
    real_write_table = pq.write_table
    calls = []

    def fail_on_second_table(table, where, *args, **kwargs):
        calls.append(where)
        if len(calls) == 2:
            raise OSError("simulated disk failure")
        return real_write_table(table, where, *args, **kwargs)

    monkeypatch.setattr(pq, "write_table", fail_on_second_table)

    with pytest.raises(OSError, match="simulated disk failure"):
        write_run(deepcopy(_RECORD), tmp_path)
    assert len(calls) == 2
    assert not list(tmp_path.glob("run-*"))


@pytest.mark.parametrize("leave_file", [False, True])
def test_write_run_refuses_an_existing_run_folder(tmp_path, monkeypatch, leave_file):
    """write_run raises FileExistsError and changes nothing when the run folder exists"""
    run_id = "f" * 32
    existing = tmp_path / f"run-{run_id}"
    existing.mkdir()
    if leave_file:
        (existing / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(hex=run_id))

    with pytest.raises(FileExistsError):
        write_run(deepcopy(_RECORD), tmp_path)
    expected = ["keep.txt"] if leave_file else []
    assert sorted(path.name for path in existing.iterdir()) == expected
    if leave_file:
        assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


# Plain text fields: only a str is accepted, so null or an integer is rejected.
_STRICT_TEXT_PATHS = [
    ("engine",),
    ("purpose",),
    ("scenario",),
    ("code_hash",),
    ("rule_hash",),
    ("lock_hash",),
    ("windows", 0, "utc_status"),
    ("sources", 0, "path"),
    ("sources", 0, "sha256"),
    ("policies", "symbol"),
    ("policies", "directions", "CALL"),
    ("policies", "directions", "PUT"),
    ("policies", "expiry"),
    ("policies", "tie_rule"),
    ("policies", "rounding"),
    ("policies", "cash_rule"),
    ("policies", "missing_side_rule"),
    ("policies", "order"),
    ("completeness", "status"),
    ("gate", "status"),
]

# Descriptive fields: any value is allowed except null, which would hide a missing value.
_NOT_NULL_PATHS = [
    ("strategy", "source"),
    ("strategy", "hash"),
    ("strategy", "parameters"),
    ("strategy", "seed"),
    ("utc_proof_references",),
    ("holdout",),
    ("quote_provenance",),
    ("information_availability",),
    ("selected_prices",),
    ("gate", "evidence"),
]

_DICT_GROUP_PATHS = [
    ("strategy",),
    ("policies",),
    ("policies", "directions"),
    ("quote_diagnostics",),
    ("counts",),
    ("completeness",),
    ("gate",),
]
_LIST_PATHS = [("windows",), ("sources",)]
_LIST_ITEM_PATHS = [("windows", 0), ("sources", 0)]

# Counts must not be negative: the four run counts and the three quote diagnostics.
_NON_NEGATIVE_COUNT_PATHS = [
    ("counts", "signals"),
    ("counts", "contracts"),
    ("counts", "skipped_signals"),
    ("counts", "unresolved_contracts"),
    ("quote_diagnostics", "skipped_rows"),
    ("quote_diagnostics", "missing_bid"),
    ("quote_diagnostics", "missing_ask"),
]


def _path_id(path):
    """Return a test id for a manifest path, such as policies_directions_CALL."""
    return "_".join(str(key) for key in path)


def _field_name(path):
    """Return a manifest path as read_run names it, such as manifest.windows[0]."""
    name = "manifest"
    for key in path:
        name += f"[{key}]" if isinstance(key, int) else f".{key}"
    return name


def _wrong_container_cases():
    """Build (path, value) pairs with a wrong container type."""
    not_a_dict = [None, [], "x", 1]
    not_a_list = [None, {}, "x", 1]
    dict_groups = [(path, value) for path in _DICT_GROUP_PATHS for value in not_a_dict]
    lists = [(path, value) for path in _LIST_PATHS for value in not_a_list]
    list_items = [(path, value) for path in _LIST_ITEM_PATHS for value in not_a_dict]
    return [
        pytest.param(path, value, id=f"{_path_id(path)}-{type(value).__name__}")
        for path, value in dict_groups + lists + list_items
    ]


@pytest.mark.parametrize("schema_version", [0, 2, -1, 1_000_000])
def test_run_with_unsupported_schema_version_is_rejected(tmp_path, schema_version):
    """a schema_version other than 1 is rejected, so a future format is never read as v1"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="schema_version"):
        read_run(run_dir)


@pytest.mark.parametrize(
    "label",
    ["", "Deriv live", "mt5 simulation", "MT5 simulation "],
    ids=["empty", "other_label", "lowercase", "trailing_space"],
)
def test_run_with_wrong_label_is_rejected(tmp_path, label):
    """a label other than exactly MT5 simulation is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["label"] = label
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="label"):
        read_run(run_dir)


@pytest.mark.parametrize("field", ["windows", "sources"])
def test_run_with_empty_windows_or_sources_is_rejected(tmp_path, field):
    """an empty windows or sources list is rejected, since the run has no time or source"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = []
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=field):
        read_run(run_dir)


@pytest.mark.parametrize("value", [None, 5], ids=["null", "integer"])
@pytest.mark.parametrize("path", _STRICT_TEXT_PATHS, ids=_path_id)
def test_run_with_non_text_value_in_text_field_is_rejected(tmp_path, path, value):
    """a null or integer in a plain text manifest field is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    parent[key] = value
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(path))):
        read_run(run_dir)


@pytest.mark.parametrize("path", _NOT_NULL_PATHS, ids=_path_id)
def test_run_with_null_in_descriptive_field_is_rejected(tmp_path, path):
    """a null descriptive manifest field is rejected, since null hides a gap"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    parent[key] = None
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(path))):
        read_run(run_dir)


@pytest.mark.parametrize("path", _NON_NEGATIVE_COUNT_PATHS, ids=_path_id)
def test_run_with_negative_count_is_rejected(tmp_path, path):
    """a negative count in the manifest is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    parent[key] = -1
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(path))):
        read_run(run_dir)


# counts.signals is the number of signals rows. counts.contracts is the number of contracts
# rows whose status is not "skipped": the contracts table also holds a row for each skipped signal.
@pytest.mark.parametrize("value", [0, 2])
@pytest.mark.parametrize("name", ["signals", "contracts"])
def test_run_with_count_that_differs_from_saved_rows_is_rejected(tmp_path, name, value):
    """a count that differs from the row count of its saved Parquet table is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"][name] = value
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(("counts", name)))):
        read_run(run_dir)


# Every saved column is required evidence: the reader rejects a table that drops a column,
# adds one, holds a null, or stores a different type (such as float money).
_TABLE_COLUMNS = [
    pytest.param(table, column, id=f"{table}.{column}")
    for table, columns in {
        "signals": (
            "signal_id",
            "direction",
            "requested_start_timestamp_ns",
            "available_timestamp_ns",
        ),
        "contracts": (
            "signal_id",
            "direction",
            "price_reference",
            "status",
            "reason",
            "outcome",
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
        "cash": (
            "event",
            "signal_id",
            "before_available_cents",
            "before_locked_cents",
            "available_cents",
            "locked_cents",
        ),
    }.items()
    for column in columns
]


def _rewrite_table(run_dir, name, change):
    path = run_dir / f"{name}.parquet"
    pq.write_table(change(pq.read_table(path)), path)


@pytest.mark.parametrize(("table", "column"), _TABLE_COLUMNS)
def test_run_with_a_missing_table_column_is_rejected(tmp_path, table, column):
    """a saved table that lacks a required column is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    _rewrite_table(run_dir, table, lambda t: t.drop_columns([column]))

    with pytest.raises(ValueError, match=re.escape(f"{table}.{column}")):
        read_run(run_dir)


@pytest.mark.parametrize(("table", "column"), _TABLE_COLUMNS)
def test_run_with_a_null_table_value_is_rejected(tmp_path, table, column):
    """a saved table with a null in a required column is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)

    def null_first(t):
        index = t.schema.get_field_index(column)
        values = t.column(column).to_pylist()
        values[0] = None
        return t.set_column(
            index, column, pa.array(values, type=t.schema.field(column).type)
        )

    _rewrite_table(run_dir, table, null_first)

    with pytest.raises(ValueError, match=re.escape(f"{table}.{column}")):
        read_run(run_dir)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        p
        for p in _TABLE_COLUMNS
        if p.values[1].endswith(("_ns", "_cents", "_ordinal", "_spot"))
    ],
)
def test_run_with_fractional_integer_column_is_rejected(tmp_path, table, column):
    """a saved money, time, ordinal or price column stored as a float is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)

    def as_fraction(t):
        index = t.schema.get_field_index(column)
        values = [value + 0.25 for value in t.column(column).to_pylist()]
        return t.set_column(index, column, pa.array(values, type=pa.float64()))

    _rewrite_table(run_dir, table, as_fraction)

    with pytest.raises(ValueError, match=re.escape(f"{table}.{column}")):
        read_run(run_dir)


@pytest.mark.parametrize("table", ["signals", "contracts", "cash"])
def test_run_with_an_extra_table_column_is_rejected(tmp_path, table):
    """a saved table with a column outside the schema is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    _rewrite_table(
        run_dir,
        table,
        lambda t: t.append_column("extra", pa.array(["x"] * t.num_rows)),
    )

    with pytest.raises(ValueError, match=re.escape(f"{table}.extra")):
        read_run(run_dir)


@pytest.mark.parametrize("name", ["skipped_signals", "unresolved_contracts"])
def test_run_with_status_count_that_differs_from_saved_rows_is_rejected(tmp_path, name):
    """a skipped or unresolved count that differs from the saved contract rows is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"][name] = 99
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(("counts", name)))):
        read_run(run_dir)


def test_run_with_a_skipped_signal_row_is_read(tmp_path):
    """a skipped signal has a signals row and a contracts row but is not a contract count"""
    record = deepcopy(_RECORD)
    record["signals"].append({**_SIGNAL_ROWS[0], "signal_id": "signal-2"})
    record["contracts"].append(
        {
            **dict.fromkeys(_CONTRACT_ROWS[0], 0),
            "signal_id": "signal-2",
            "direction": "CALL",
            "price_reference": "bid",
            "status": "skipped",
            "reason": "five contracts open",
            "outcome": "",
        }
    )
    record["manifest"]["counts"].update(signals=2, contracts=1, skipped_signals=1)
    run_dir = write_run(record, tmp_path)

    run = read_run(run_dir)
    assert len(run["signals"]) == 2
    assert len(run["contracts"]) == 2
    assert run["manifest"]["counts"]["contracts"] == 1


@pytest.mark.parametrize(("path", "value"), _wrong_container_cases())
def test_run_with_wrong_container_type_is_rejected(tmp_path, path, value):
    """a group, list or list item with the wrong container type is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parent, key = _parent_and_key(manifest, path)
    parent[key] = value
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape(_field_name(path))):
        read_run(run_dir)


@pytest.mark.parametrize(
    ("list_name", "break_item"),
    [
        pytest.param(
            "windows",
            lambda item: item.pop("utc_offset_seconds"),
            id="windows_1_missing_utc_offset_seconds",
        ),
        pytest.param(
            "sources",
            lambda item: item.update(price_scale=float(item["price_scale"])),
            id="sources_1_float_price_scale",
        ),
    ],
)
def test_run_checks_every_list_item_not_only_the_first(tmp_path, list_name, break_item):
    """a broken second list item is rejected, not only the first item"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    second_item = deepcopy(manifest[list_name][0])
    break_item(second_item)
    manifest[list_name].append(second_item)
    _save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match=re.escape("[1]")):
        read_run(run_dir)


@pytest.mark.parametrize("content", [None, [1]], ids=["null", "list"])
def test_run_with_non_object_manifest_is_rejected(tmp_path, content):
    """a manifest file whose JSON is not an object is rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    _save_manifest(run_dir, content)

    with pytest.raises(ValueError, match="manifest must be an object"):
        read_run(run_dir)


def test_run_with_extra_unknown_keys_is_read(tmp_path):
    """an extra key at top level or in policies is read and kept, not rejected"""
    run_dir = write_run(deepcopy(_RECORD), tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["future_note"] = "top-level extra key"
    manifest["policies"]["future_note"] = "policy extra key"
    _save_manifest(run_dir, manifest)

    run = read_run(run_dir)
    assert set(run) == {"manifest", "signals", "contracts", "cash", "summary"}
    assert run["manifest"]["future_note"] == "top-level extra key"
    assert run["manifest"]["policies"]["future_note"] == "policy extra key"
