"""A-share symbol normalization and trading-rule context.

The rules here are deterministic prompt guidance, not a legal manual. They keep
agents from applying US-market assumptions to mainland China stocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import re


class AShareBoard(str, Enum):
    SH_MAIN = "Shanghai Main Board"
    STAR = "STAR Market"
    SZ_MAIN = "Shenzhen Main Board"
    CHINEXT = "ChiNext"
    BSE = "Beijing Stock Exchange"


_A_SHARE_RE = re.compile(r"^(?P<code>\d{6})(?P<suffix>\.(SS|SZ|BJ))?$", re.IGNORECASE)


def _parse_code(symbol: str) -> tuple[str, str | None]:
    if not isinstance(symbol, str):
        raise ValueError(f"{symbol!r} is not a supported A-share symbol")
    match = _A_SHARE_RE.fullmatch(symbol.strip().upper())
    if not match:
        raise ValueError(f"{symbol!r} is not a supported A-share symbol")
    return match.group("code"), match.group("suffix")


def _board_for_code(code: str) -> AShareBoard:
    if code.startswith(("600", "601", "603", "605")):
        return AShareBoard.SH_MAIN
    if code.startswith(("688", "689")):
        return AShareBoard.STAR
    if code.startswith(("000", "001", "002", "003")):
        return AShareBoard.SZ_MAIN
    if code.startswith(("300", "301")):
        return AShareBoard.CHINEXT
    if code.startswith(("8", "4", "9")):
        return AShareBoard.BSE
    raise ValueError(f"{code!r} is not a supported A-share symbol")


def _suffix_for_board(board: AShareBoard) -> str:
    if board in (AShareBoard.SH_MAIN, AShareBoard.STAR):
        return ".SS"
    if board in (AShareBoard.SZ_MAIN, AShareBoard.CHINEXT):
        return ".SZ"
    return ".BJ"


def is_a_share_symbol(symbol: str) -> bool:
    try:
        code, _ = _parse_code(symbol)
        _board_for_code(code)
    except ValueError:
        return False
    return True


def classify_a_share_board(symbol: str) -> AShareBoard:
    code, _ = _parse_code(symbol)
    return _board_for_code(code)


def normalize_a_share_symbol(symbol: str) -> str:
    code, suffix = _parse_code(symbol)
    board = _board_for_code(code)
    return f"{code}{suffix or _suffix_for_board(board)}".upper()


def _price_limit_text(board: AShareBoard) -> str:
    if board in (AShareBoard.SH_MAIN, AShareBoard.SZ_MAIN):
        return "Main-board stocks generally have a 10% daily price limit; ST/*ST risk-warning stocks generally have a 5% daily price limit."
    if board in (AShareBoard.STAR, AShareBoard.CHINEXT):
        return f"{board.value} stocks generally have a 20% daily price limit after the initial no-limit listing period."
    return "Beijing Stock Exchange stocks generally have a 30% daily price limit after the initial no-limit listing period."


def build_a_share_rule_context(
    symbol: str,
    trade_date: str | None = None,
    identity: Mapping[str, str] | None = None,
) -> str:
    board = classify_a_share_board(symbol)
    normalized = normalize_a_share_symbol(symbol)
    name = (identity or {}).get("company_name") or (identity or {}).get("name")
    name_part = f" Company/name context: {name}." if name else ""
    date_part = f" Analysis date: {trade_date}." if trade_date else ""
    lot_rule = (
        "STAR Market buy orders commonly start at 200 shares, then allow one-share increments where applicable."
        if board is AShareBoard.STAR
        else "Standard board-lot trading is 100 shares per lot."
    )
    return (
        f"A-share trading constraints for `{normalized}` ({board.value})."
        f"{name_part}{date_part} Trading currency is CNY. Settlement is T+1 for stocks, "
        "so do not assume US-style intraday sell-after-buy. "
        f"{lot_rule} "
        "Regular trading includes opening call auction, continuous auction, a midday break, "
        "and closing call auction; avoid assuming an uninterrupted US-style session. "
        f"{_price_limit_text(board)} "
        "New listings may have no daily price limit during the first five trading days, depending on board rules. "
        "limit-up or limit-down states can make a BUY/SELL recommendation unexecutable even when the nominal signal is clear. "
        "Check suspensions, ex-rights/ex-dividend adjustments, disclosure events, and regulatory risk warnings before making strong price-series claims."
    )
