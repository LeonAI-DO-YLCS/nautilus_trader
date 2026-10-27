"""
Cash and contract ledger for Deriv binary options over a tick stream.

`run_ledger` turns plain ints into plain int arrays and runs one loop over them. The loop keeps
all state in ints and int arrays and calls only the shared Deriv rules, so it can be compiled
with Numba. There are no floats. Prices stay in the input half-cent units: the ledger
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
from numba import njit
from numba.extending import register_jitable

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
    _WINDOW_END,
) = range(9)
_REASON_TEXTS = (
    "",
    "settled at expiry on a fresh last price",
    "capacity limit: five contracts already open",
    "insufficient cash: available cash is below the minimum stake",
    "first eligible tick is after expiry",
    "no eligible tick before the data ended",
    "last price before expiry is more than 10 seconds old, stake stays locked",
    "data ended before expiry, stake stays locked",
    "first eligible tick is at or after the window end",
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


@dataclass
class _LedgerState:
    contracts: np.ndarray
    cash_log: np.ndarray
    pending: np.ndarray
    slots: np.ndarray
    n_pending: int
    n_slots: int
    next_start: int
    available: int
    locked: int
    cash_rows: int
    tail_group: list[tuple[int, int, int, int, int]]
    previous_row: tuple[int, int, int, int, int] | None
    last_timestamp: int | None
    ended_open: int = 0


def _validate_run_options(scenario, starting_cash_cents, window_end_ns, compiled):
    if scenario not in _SCENARIO_COLUMNS:
        raise ValueError("scenario must be bid, mid or ask")
    if type(compiled) is not bool:
        raise TypeError("compiled must be a bool")
    checked_cents(starting_cash_cents)
    if window_end_ns is not None:
        checked_add(window_end_ns, 0)


def _new_core_state(n_signals, starting_cash):
    contracts = np.zeros((n_signals, _CONTRACT_WIDTH), dtype=np.int64)
    cash_log = np.zeros((1 + 2 * n_signals, _CASH_WIDTH), dtype=np.int64)
    _write_cash(cash_log, 0, _INITIAL, -1, 0, 0, starting_cash, 0)
    return _LedgerState(
        contracts=contracts,
        cash_log=cash_log,
        pending=np.zeros(n_signals, dtype=np.int64),
        slots=np.zeros(_MAX_OPEN, dtype=np.int64),
        n_pending=0,
        n_slots=0,
        next_start=0,
        available=starting_cash,
        locked=0,
        cash_rows=1,
        tail_group=[],
        previous_row=None,
        last_timestamp=None,
    )


def run_ledger(
    ticks: list[tuple[int, int, int, int, int]],
    signals: list[dict[str, object]],
    *,
    scenario: str,
    starting_cash_cents: int,
    window_end_ns: int | None = None,
    compiled: bool = False,
) -> LedgerResult:
    """
    Run the cash and contract ledger for one price scenario.

    `ticks` are `(timestamp_ns, ordinal, bid, mid, ask)` ints in source order and `signals`
    are dicts in frozen order. `scenario` ("bid", "mid" or "ask") picks the price column.
    """
    _validate_run_options(scenario, starting_cash_cents, window_end_ns, compiled)
    tick_table = _tick_table(ticks)
    timestamps = np.ascontiguousarray(tick_table[:, 0])
    if (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError("tick timestamps must not decrease")
    starts, availabilities, is_call = _signal_arrays(signals)
    by_start = np.argsort(starts, kind="stable").astype(np.int64)
    state = _run_core(
        timestamps,
        np.ascontiguousarray(tick_table[:, 1]),
        np.ascontiguousarray(tick_table[:, _SCENARIO_COLUMNS[scenario]]),
        starts,
        availabilities,
        is_call,
        by_start,
        starting_cash_cents,
        window_end_ns,
        compiled=compiled,
    )
    return _build_result(
        signals,
        scenario,
        state.contracts,
        state.cash_log[: state.cash_rows],
        state.ended_open,
    )


def _process_chunk_group(
    state,
    group_rows,
    previous_row,
    starts,
    availabilities,
    is_call,
    by_start,
    starting_cash,
    window_end_ns,
    price_column,
    compiled=False,
):
    rows = [*([previous_row] if previous_row is not None else []), *group_rows]
    table = _tick_table(rows)
    timestamps = np.ascontiguousarray(table[:, 0])
    return _run_core(
        timestamps,
        np.ascontiguousarray(table[:, 1]),
        np.ascontiguousarray(table[:, price_column]),
        starts,
        availabilities,
        is_call,
        by_start,
        starting_cash,
        window_end_ns,
        state=state,
        start_index=1 if previous_row is not None else 0,
        finalize=False,
        compiled=compiled,
    )


def run_ledger_chunks(
    chunks,
    signals: list[dict[str, object]],
    *,
    scenario: str,
    starting_cash_cents: int,
    window_end_ns: int | None = None,
    compiled: bool = False,
) -> LedgerResult:
    """
    Run the ledger incrementally while retaining only one incomplete timestamp group.

    A group is processed when a later timestamp completes it. The previous group's last row
    remains available for expiry selection. The final tail group is processed before end-of-data
    handling.
    """
    _validate_run_options(scenario, starting_cash_cents, window_end_ns, compiled)
    starts, availabilities, is_call = _signal_arrays(signals)
    by_start = np.argsort(starts, kind="stable").astype(np.int64)
    state = _new_core_state(len(signals), starting_cash_cents)
    price_column = _SCENARIO_COLUMNS[scenario]

    for chunk in chunks:
        table = _tick_table(chunk)
        for row_index in range(table.shape[0]):
            row = tuple(int(value) for value in table[row_index])
            if state.last_timestamp is not None and row[0] < state.last_timestamp:
                raise ValueError("tick timestamps must not decrease")
            if state.tail_group and row[0] != state.tail_group[-1][0]:
                state = _process_chunk_group(
                    state,
                    state.tail_group,
                    state.previous_row,
                    starts,
                    availabilities,
                    is_call,
                    by_start,
                    starting_cash_cents,
                    window_end_ns,
                    price_column,
                    compiled=compiled,
                )
                state.previous_row = state.tail_group[-1]
                state.tail_group.clear()
            state.tail_group.append(row)
            state.last_timestamp = row[0]

    if state.tail_group:
        state = _process_chunk_group(
            state,
            state.tail_group,
            state.previous_row,
            starts,
            availabilities,
            is_call,
            by_start,
            starting_cash_cents,
            window_end_ns,
            price_column,
            compiled=compiled,
        )
        state.previous_row = state.tail_group[-1]
        state.tail_group.clear()

    empty = np.empty(0, dtype=np.int64)
    state = _run_core(
        empty,
        empty,
        empty,
        starts,
        availabilities,
        is_call,
        by_start,
        starting_cash_cents,
        window_end_ns,
        state=state,
        finalize=True,
        compiled=compiled,
    )
    return _build_result(
        signals,
        scenario,
        state.contracts,
        state.cash_log[: state.cash_rows],
        state.ended_open,
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
    if exit_tick < 0:
        contracts[index, _STATUS] = _UNRESOLVED
        contracts[index, _REASON] = _STALE_EXIT
        return cash_rows, available, locked
    exit_ns = int(timestamps[exit_tick])
    if (exit_ns < 0 and expiry > (2**63 - 1) + exit_ns) or not exit_is_fresh(
        exit_ns, expiry
    ):
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


def _run_core_loop(
    timestamps,
    ordinals,
    prices,
    starts,
    availabilities,
    is_call,
    by_start,
    contracts,
    cash_log,
    pending,
    slots,
    n_pending,
    n_slots,
    next_start,
    available,
    locked,
    cash_rows,
    ended_open,
    has_window_end,
    window_end_ns,
    start_index,
    finalize,
):
    """Process complete timestamp groups using arrays and scalar state only."""
    n_ticks = timestamps.shape[0]
    n_signals = starts.shape[0]
    group_start = start_index
    while group_start < n_ticks:
        group_ns = timestamps[group_start]
        group_end = group_start + 1
        while group_end < n_ticks and timestamps[group_end] == group_ns:
            group_end += 1

        exit_begin = group_start - 1 if group_start > 0 else 0

        # Settle due contracts in admission order.
        kept = 0
        for position in range(n_slots):
            index = slots[position]
            expiry = contracts[index, _EXPIRY_NS]
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
            if has_window_end and starts[index] >= window_end_ns:
                _skip(contracts, index, _WINDOW_END)
                continue
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
                if has_window_end and group_ns >= window_end_ns:
                    _skip(contracts, index, _WINDOW_END)
                elif group_ns > expiry:
                    _skip(contracts, index, _ENTRY_LATE)
                else:
                    pending[kept] = index
                    kept += 1
                continue
            if has_window_end and timestamps[entry_tick] >= window_end_ns:
                _skip(contracts, index, _WINDOW_END)
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
            contracts[index, _REASON] = _DATA_ENDED
            contracts[index, _ENTRY_ORDINAL] = ordinals[entry_tick]
            contracts[index, _ENTRY_NS] = timestamps[entry_tick]
            contracts[index, _EXPIRY_NS] = expiry
            contracts[index, _ENTRY_SPOT] = prices[entry_tick]
            contracts[index, _STAKE] = stake
            slots[n_slots] = index
            n_slots += 1
        n_pending = kept

        # Settle contracts admitted at their expiry on this group's last source row.
        kept = 0
        for position in range(n_slots):
            index = slots[position]
            expiry = contracts[index, _EXPIRY_NS]
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

    if finalize:
        ended_open = 0
        for index in range(n_signals):
            if contracts[index, _STATUS] == _OPEN:
                contracts[index, _STATUS] = _UNRESOLVED
                contracts[index, _REASON] = _DATA_ENDED
                ended_open += 1
            elif contracts[index, _STATUS] == _WAITING:
                reason = (
                    _WINDOW_END
                    if has_window_end and starts[index] >= window_end_ns
                    else _NO_ENTRY_DATA
                )
                _skip(contracts, index, reason)
    return (
        n_pending,
        n_slots,
        next_start,
        available,
        locked,
        cash_rows,
        ended_open,
    )


def _run_core(
    timestamps,
    ordinals,
    prices,
    starts,
    availabilities,
    is_call,
    by_start,
    starting_cash,
    window_end_ns,
    *,
    state=None,
    start_index=0,
    finalize=True,
    compiled=False,
):
    """Run the shared ledger loop with Python or Numba execution."""
    if state is None:
        state = _new_core_state(starts.shape[0], starting_cash)
    has_window_end = window_end_ns is not None
    end_value = window_end_ns if has_window_end else 0
    core = _compiled_core_loop if compiled else _run_core_loop
    (
        state.n_pending,
        state.n_slots,
        state.next_start,
        state.available,
        state.locked,
        state.cash_rows,
        state.ended_open,
    ) = core(
        timestamps,
        ordinals,
        prices,
        starts,
        availabilities,
        is_call,
        by_start,
        state.contracts,
        state.cash_log,
        state.pending,
        state.slots,
        state.n_pending,
        state.n_slots,
        state.next_start,
        state.available,
        state.locked,
        state.cash_rows,
        state.ended_open,
        has_window_end,
        end_value,
        start_index,
        finalize,
    )
    return state


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


# Register original shared rules and selectors as Numba-jitable; the Deriv package stays stdlib-only.
for _jitable in (
    checked_cents,
    checked_add,
    stake_cents,
    gross_credit_cents,
    is_win,
    expiry_ns,
    entry_index_in_range,
    exit_index_in_range,
    exit_is_fresh,
    _write_cash,
    _skip,
    _settle,
    _run_core_loop,
):
    register_jitable(_jitable)

_compiled_core_loop = njit(fastmath=False)(_run_core_loop)
