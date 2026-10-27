"""Exact-cent stake, payout and outcome rules for the Deriv binary options profile.

Amounts are integer cents. Each public function checks its numeric inputs first: floats and
bools raise TypeError, and values outside the signed 64-bit range raise OverflowError.

Exception messages are constant strings so Numba can compile these same functions. The package
stays stdlib-only: the vectorbt adapter registers `checked_cents` with Numba's `register_jitable`
and compiles the public functions with `fastmath=False`.
"""

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MIN_STAKE_CENTS = 35
_WIN_MULTIPLIER_THOUSANDTHS = (
    1952  # gross credit on a win = stake * 1.952 (95.2% net payout)
)
_MAX_WIN_STAKE_CENTS = (
    _INT64_MAX // _WIN_MULTIPLIER_THOUSANDTHS
)  # largest |stake| whose win credit fits


def stake_cents(available_cents: int) -> int | None:
    """Return the stake for a run: 0.05% of available cash, rounded down, at least 0.35 USD.

    Return None when available cash cannot pay the minimum stake: no trade.
    """
    checked_cents(available_cents)
    if available_cents < _MIN_STAKE_CENTS:
        return None
    return max(_MIN_STAKE_CENTS, available_cents // 2000)


def gross_credit_cents(stake_cents: int, won: bool) -> int:
    """Return the cash credited at settlement: stake * 1.952 on a win, 0 on a loss or tie."""
    checked_cents(stake_cents)
    if not won:
        return 0
    if not -_MAX_WIN_STAKE_CENTS <= stake_cents <= _MAX_WIN_STAKE_CENTS:
        raise OverflowError("stake cents: win credit overflows signed 64-bit")
    return stake_cents * _WIN_MULTIPLIER_THOUSANDTHS // 1000


def is_win(direction: str, entry_cents: int, exit_cents: int) -> bool:
    """Return True when the contract wins: CALL (Rise) needs exit above entry, PUT (Fall) below.

    A tie loses for both directions. Any other direction raises ValueError.
    """
    checked_cents(entry_cents)
    checked_cents(exit_cents)
    if direction == "CALL":
        return exit_cents > entry_cents
    if direction == "PUT":
        return exit_cents < entry_cents
    raise ValueError("unknown direction: expected CALL or PUT")


def checked_cents(value: int) -> int:
    """Return value if it is an int in the signed 64-bit range.

    Raise TypeError for non-int values (floats and bools included) and OverflowError
    for values outside the range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cents must be int")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise OverflowError("cents value is outside the signed 64-bit range")
    return value


def checked_add(a: int, b: int) -> int:
    """Return a + b, checking both operands and the sum; raise OverflowError instead of wrapping."""
    checked_cents(a)
    checked_cents(b)
    if (b > 0 and a > _INT64_MAX - b) or (b < 0 and a < _INT64_MIN - b):
        raise OverflowError("sum is outside the signed 64-bit range")
    return checked_cents(a + b)
