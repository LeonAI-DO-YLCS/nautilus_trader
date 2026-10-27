"""
Cash and contract ledger for Deriv binary options over a tick stream.

`run_ledger` turns plain ints into plain int arrays and runs one loop over them. The loop keeps
all state in ints and int arrays and calls only the shared Deriv rules, so it can be compiled
with Numba later. There are no floats. Prices stay in the input half-cent units: the ledger
compares them and copies them, and never converts them.

Per timestamp group the loop does four things in this order: it consumes every tick of the
group, settles the open contracts that are due (in admission order), admits entries (in
signal-list order), then settles the contracts just admitted whose expiry is this group (in
admission order). A contract that cannot be settled keeps its stake locked and its slot.

Entry and exit rows come from the shared selectors in `deriv_mt5.ticks`, and only over the rows
of the current group (plus the previous source row for a due exit), never over the whole history.
"""

from dataclasses import dataclass
from itertools import chain

import numpy as np
from deriv_mt5.rules import (
    checked_add,
    checked_cents,
    gross_credit_cents,
    is_win,
    stake_cents,
)
from deriv_mt5.ticks import (
    entry_index_in_range,
    exit_index_in_range,
    exit_is_fresh,
    expiry_ns,
)

_MAX_OPEN = 5
_SCENARIO_COLUMNS = {"bid": 2, "mid": 3, "ask": 4}  # column in a tick row
_DIRECTIONS = ("CALL", "PUT")

# Contract states. The first two never reach the result: the last sweep changes them.
(_WAITING, _OPEN, _SETTLED, _SKIPPED, _UNRESOLVED) = range(5)
_STATUS_NAMES = ("waiting", "open", "settled", "skipped", "unresolved")

# Reasons, by code. Wording is not approved yet.
(
    _NO_REASON,
    _SETTLED_FRESH,
    _FULL,
    _NO_CASH,
    _ENTRY_LATE,
    _NO_ENTRY_DATA,
    _STALE_EXIT,
    _DATA_ENDED,
) = range(8)
_REASON_TEXTS = (
    "",
    "settled at expiry on a fresh last price",
    "five contracts already open",
    "available cash is below the minimum stake",
    "first eligible tick is after expiry",
    "no eligible tick before the data ended",
    "last price before expiry is more than 10 seconds old, stake stays locked",
    "data ended before expiry, stake stays locked",
)

_OUTCOME_NAMES = ("", "win", "loss")

# Columns of the contract table: one row per signal.
(
    _STATUS,
    _REASON,
    _OUTCOME,
    _ENTRY_ORDINAL,
    _EXIT_ORDINAL,
    _ENTRY_NS,
    _EXPIRY_NS,
    _EXIT_NS,
    _ENTRY_SPOT,
    _EXIT_SPOT,
    _STAKE,
    _GROSS,
    _NET,
    _CONTRACT_WIDTH,
) = range(14)

# Columns of the cash table: one row per cash change.
(
    _CASH_EVENT,
    _CASH_SIGNAL,
    _CASH_BEFORE_AVAILABLE,
    _CASH_BEFORE_LOCKED,
    _CASH_AVAILABLE,
    _CASH_LOCKED,
    _CASH_WIDTH,
) = range(7)
_INITIAL, _ENTRY, _SETTLEMENT = range(3)
_EVENT_NAMES = ("initial", "entry", "settlement")


@dataclass(frozen=True)
class LedgerResult:
    """
    The outcome of one ledger run: a row per signal, a row per cash change, and counts.
    """

    contracts: list[dict[str, int | str]]
    cash: list[dict[str, int | str]]
    skipped_count: int
    unresolved_count: int
    incomplete: bool
    incomplete_reason: str


def run_ledger(
    ticks: list[tuple[int, int, int, int, int]],
    signals: list[dict[str, object]],
    *,
    scenario: str,
    starting_cash_cents: int,
) -> LedgerResult:
    """
    Run the cash and contract ledger for one price scenario.

    `ticks` are `(timestamp_ns, ordinal, bid, mid, ask)` ints in source order and `signals`
    are dicts in frozen order. `scenario` ("bid", "mid" or "ask") picks the price column.
    """
    if scenario not in _SCENARIO_COLUMNS:
        raise ValueError("scenario must be bid, mid or ask")
    checked_cents(starting_cash_cents)
    tick_table = _tick_table(ticks)
    timestamps = np.ascontiguousarray(tick_table[:, 0])
    if (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError("tick timestamps must not decrease")
    starts, availabilities, is_call = _signal_arrays(signals)
    contracts, cash_log, cash_rows, ended_open = _run_core(
        timestamps,
        np.ascontiguousarray(tick_table[:, 1]),
        np.ascontiguousarray(tick_table[:, _SCENARIO_COLUMNS[scenario]]),
        starts,
        availabilities,
        is_call,
        np.argsort(starts, kind="stable").astype(np.int64),
        starting_cash_cents,
    )
    return _build_result(
        signals, scenario, contracts, cash_log[:cash_rows], int(ended_open)
    )


def _tick_table(ticks: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    """
    Return the ticks as an int64 table with five columns, or raise for floats and bad shapes.
    """
    if len(ticks) == 0:
        return np.zeros((0, 5), dtype=np.int64)
    # Check the Python values before NumPy turns them into one dtype: it would read a bool as 1.
    if set(map(type, chain.from_iterable(ticks))) != {int}:
        raise TypeError("ticks must hold plain ints inside the signed 64-bit range")
    if set(map(len, ticks)) != {5}:
        raise ValueError("each tick must be (timestamp_ns, ordinal, bid, mid, ask)")
    table = np.asarray(ticks)
    if table.dtype != np.int64:
        raise TypeError("ticks must hold plain ints inside the signed 64-bit range")
    return table


def _signal_arrays(
    signals: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return start times, availability ordinals and a 1-for-CALL flag, each as an int64 array.
    """
    for signal in signals:
        if signal["direction"] not in _DIRECTIONS:
            raise ValueError("direction must be CALL or PUT")
        for key in ("start_time_ns", "availability_ordinal"):
            if type(signal[key]) is not int:
                raise TypeError(f"{key} must be an int")
        # Raises OverflowError before any money moves when start plus 60 s leaves int64.
        expiry_ns(signal["start_time_ns"])
    return (
        np.array([signal["start_time_ns"] for signal in signals], dtype=np.int64),
        np.array(
            [signal["availability_ordinal"] for signal in signals], dtype=np.int64
        ),
        np.array([signal["direction"] == "CALL" for signal in signals], dtype=np.int64),
    )


def _write_cash(
    cash_log, row, event, signal, before_available, before_locked, available, locked
):
    """
    Write one cash change at `row` of the cash table.
    """
    cash_log[row, _CASH_EVENT] = event
    cash_log[row, _CASH_SIGNAL] = signal
    cash_log[row, _CASH_BEFORE_AVAILABLE] = before_available
    cash_log[row, _CASH_BEFORE_LOCKED] = before_locked
    cash_log[row, _CASH_AVAILABLE] = available
    cash_log[row, _CASH_LOCKED] = locked


def _skip(contracts, index, reason):
    """
    Mark a contract as skipped: no cash moves and no slot is used.
    """
    contracts[index, _STATUS] = _SKIPPED
    contracts[index, _REASON] = reason


def _settle(
    contracts,
    cash_log,
    cash_rows,
    index,
    exit_tick,
    timestamps,
    ordinals,
    prices,
    is_call,
    available,
    locked,
):
    """
    Settle the open contract `index` on row `exit_tick`, or mark it unresolved.

    A missing or stale exit row (`exit_tick` below 0 or not fresh) leaves the stake locked.
    Return the cash row count, the available cash and the locked cash after the change.
    """
    expiry = int(contracts[index, _EXPIRY_NS])
    if exit_tick < 0 or not exit_is_fresh(int(timestamps[exit_tick]), expiry):
        contracts[index, _STATUS] = _UNRESOLVED
        contracts[index, _REASON] = _STALE_EXIT
        return cash_rows, available, locked
    stake = int(contracts[index, _STAKE])
    exit_spot = int(prices[exit_tick])
    direction = "CALL" if is_call[index] == 1 else "PUT"
    won = is_win(direction, int(contracts[index, _ENTRY_SPOT]), exit_spot)
    credit = gross_credit_cents(stake, won)
    new_available = checked_add(available, credit)
    new_locked = checked_add(locked, -stake)
    _write_cash(
        cash_log,
        cash_rows,
        _SETTLEMENT,
        index,
        available,
        locked,
        new_available,
        new_locked,
    )
    contracts[index, _STATUS] = _SETTLED
    contracts[index, _REASON] = _SETTLED_FRESH
    contracts[index, _OUTCOME] = 1 if won else 2
    contracts[index, _EXIT_ORDINAL] = ordinals[exit_tick]
    contracts[index, _EXIT_NS] = timestamps[exit_tick]
    contracts[index, _EXIT_SPOT] = exit_spot
    contracts[index, _GROSS] = credit
    contracts[index, _NET] = credit - stake
    return cash_rows + 1, new_available, new_locked


def _run_core(
    timestamps,
    ordinals,
    prices,
    starts,
    availabilities,
    is_call,
    by_start,
    starting_cash,
):
    """
    Walk the ticks once, one timestamp group at a time, and fill the contract and cash tables.

    Return the contract table, the cash table, the number of cash rows used, and the number
    of contracts still open when the data ended.
    """
    n_ticks = timestamps.shape[0]
    n_signals = starts.shape[0]
    contracts = np.zeros((n_signals, _CONTRACT_WIDTH), dtype=np.int64)
    cash_log = np.zeros((1 + 2 * n_signals, _CASH_WIDTH), dtype=np.int64)
    pending = np.zeros(
        n_signals, dtype=np.int64
    )  # started, not yet entered; list order
    slots = np.zeros(_MAX_OPEN, dtype=np.int64)  # open or unresolved; admission order
    n_pending = 0
    n_slots = 0
    next_start = 0
    available = starting_cash
    locked = 0
    _write_cash(cash_log, 0, _INITIAL, -1, 0, 0, available, locked)
    cash_rows = 1

    group_start = 0
    while group_start < n_ticks:
        group_ns = timestamps[group_start]
        group_end = group_start + 1
        while group_end < n_ticks and timestamps[group_end] == group_ns:
            group_end += 1

        # A due exit looks at the previous source row (when there is one) and this group.
        exit_begin = group_start - 1 if group_start > 0 else 0

        # Settle due contracts in admission order.
        kept = 0
        for position in range(n_slots):
            index = slots[position]
            expiry = int(contracts[index, _EXPIRY_NS])
            if contracts[index, _STATUS] != _OPEN or expiry > group_ns:
                slots[kept] = index
                kept += 1
                continue
            exit_tick = exit_index_in_range(timestamps, exit_begin, group_end, expiry)
            cash_rows, available, locked = _settle(
                contracts,
                cash_log,
                cash_rows,
                index,
                exit_tick,
                timestamps,
                ordinals,
                prices,
                is_call,
                available,
                locked,
            )
            if contracts[index, _STATUS] != _SETTLED:
                slots[kept] = index
                kept += 1
        n_slots = kept

        # Signals whose start time has been reached wait here, kept in list order.
        while next_start < n_signals and starts[by_start[next_start]] <= group_ns:
            index = by_start[next_start]
            position = n_pending
            while position > 0 and pending[position - 1] > index:
                pending[position] = pending[position - 1]
                position -= 1
            pending[position] = index
            n_pending += 1
            next_start += 1

        # Admit entries in signal-list order.
        kept = 0
        for position in range(n_pending):
            index = pending[position]
            expiry = expiry_ns(int(starts[index]))
            entry_tick = entry_index_in_range(
                timestamps,
                ordinals,
                group_start,
                group_end,
                int(starts[index]),
                int(availabilities[index]),
            )
            if entry_tick < 0:
                if group_ns > expiry:  # no row of this group can be in time: too late
                    _skip(contracts, index, _ENTRY_LATE)
                else:  # rows are in time but none is available yet
                    pending[kept] = index
                    kept += 1
                continue
            if n_slots >= _MAX_OPEN:
                _skip(contracts, index, _FULL)
                continue
            maybe_stake = stake_cents(available)
            if maybe_stake is None:
                _skip(contracts, index, _NO_CASH)
                continue
            stake = int(maybe_stake)
            new_available = checked_add(available, -stake)
            new_locked = checked_add(locked, stake)
            _write_cash(
                cash_log,
                cash_rows,
                _ENTRY,
                index,
                available,
                locked,
                new_available,
                new_locked,
            )
            cash_rows += 1
            available = new_available
            locked = new_locked
            contracts[index, _STATUS] = _OPEN
            contracts[index, _REASON] = _DATA_ENDED  # replaced when it settles
            contracts[index, _ENTRY_ORDINAL] = ordinals[entry_tick]
            contracts[index, _ENTRY_NS] = timestamps[entry_tick]
            contracts[index, _EXPIRY_NS] = expiry
            contracts[index, _ENTRY_SPOT] = prices[entry_tick]
            contracts[index, _STAKE] = stake
            slots[n_slots] = index
            n_slots += 1
        n_pending = kept

        # A contract admitted at its own expiry settles now on this group's last row, with no
        # later tick needed. Admission is not run again, so skipped signals stay skipped.
        kept = 0
        for position in range(n_slots):
            index = slots[position]
            expiry = int(contracts[index, _EXPIRY_NS])
            if contracts[index, _STATUS] == _OPEN and expiry <= group_ns:
                exit_tick = exit_index_in_range(
                    timestamps, exit_begin, group_end, expiry
                )
                cash_rows, available, locked = _settle(
                    contracts,
                    cash_log,
                    cash_rows,
                    index,
                    exit_tick,
                    timestamps,
                    ordinals,
                    prices,
                    is_call,
                    available,
                    locked,
                )
                if contracts[index, _STATUS] == _SETTLED:
                    continue
            slots[kept] = index
            kept += 1
        n_slots = kept

        group_start = group_end

    # The data is over: what is still open keeps its stake locked, what never entered is skipped.
    ended_open = 0
    for index in range(n_signals):
        if contracts[index, _STATUS] == _OPEN:
            contracts[index, _STATUS] = _UNRESOLVED
            contracts[index, _REASON] = _DATA_ENDED
            ended_open += 1
        elif contracts[index, _STATUS] == _WAITING:
            _skip(contracts, index, _NO_ENTRY_DATA)
    return contracts, cash_log, cash_rows, ended_open


def _build_result(
    signals: list[dict[str, object]],
    scenario: str,
    contract_table: np.ndarray,
    cash_table: np.ndarray,
    ended_open: int,
) -> LedgerResult:
    """
    Turn the integer tables into the dict rows of a `LedgerResult`.
    """
    contracts = []
    for signal, row in zip(signals, contract_table.tolist(), strict=True):
        contracts.append(
            {
                "signal_id": signal["signal_id"],
                "direction": signal["direction"],
                "price_reference": scenario,
                "status": _STATUS_NAMES[row[_STATUS]],
                "reason": _REASON_TEXTS[row[_REASON]],
                "outcome": _OUTCOME_NAMES[row[_OUTCOME]],
                "entry_ordinal": row[_ENTRY_ORDINAL],
                "exit_ordinal": row[_EXIT_ORDINAL],
                "entry_timestamp_ns": row[_ENTRY_NS],
                "expiry_timestamp_ns": row[_EXPIRY_NS],
                "exit_timestamp_ns": row[_EXIT_NS],
                "entry_spot": row[_ENTRY_SPOT],
                "exit_spot": row[_EXIT_SPOT],
                "stake_cents": row[_STAKE],
                "gross_credit_cents": row[_GROSS],
                "net_cents": row[_NET],
            }
        )
    cash = [
        {
            "event": _EVENT_NAMES[row[_CASH_EVENT]],
            "signal_id": ""
            if row[_CASH_SIGNAL] < 0
            else signals[row[_CASH_SIGNAL]]["signal_id"],
            "before_available_cents": row[_CASH_BEFORE_AVAILABLE],
            "before_locked_cents": row[_CASH_BEFORE_LOCKED],
            "available_cents": row[_CASH_AVAILABLE],
            "locked_cents": row[_CASH_LOCKED],
        }
        for row in cash_table.tolist()
    ]
    return LedgerResult(
        contracts=contracts,
        cash=cash,
        skipped_count=sum(row["status"] == "skipped" for row in contracts),
        unresolved_count=sum(row["status"] == "unresolved" for row in contracts),
        incomplete=ended_open > 0,
        incomplete_reason=(
            f"data ended with {ended_open} contract(s) still open" if ended_open else ""
        ),
    )
