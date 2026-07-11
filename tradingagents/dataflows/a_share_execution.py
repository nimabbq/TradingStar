"""Deterministic, account-aware A-share execution envelope."""

from __future__ import annotations

import math

import pandas as pd

from .a_share_rules import AShareBoard, classify_a_share_board, is_a_share_symbol, normalize_a_share_symbol
from .akshare import _stock_history_frame


_TARGET_WEIGHTS = {
    "buy": 0.50,
    "overweight": 0.35,
    "hold": None,
    "underweight": 0.15,
    "sell": 0.0,
}


def _latest_raw_price(ticker: str, curr_date: str) -> tuple[float, str]:
    end = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(end):
        raise ValueError(f"invalid execution date {curr_date!r}")
    _, frame = _stock_history_frame(
        ticker,
        (end - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        curr_date,
        adjust="",
    )
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    if frame.empty:
        raise ValueError(f"no unadjusted execution price for {ticker} on or before {curr_date}")
    latest = frame.iloc[-1]
    return float(latest["Close"]), latest["Date"].strftime("%Y-%m-%d")


def _buy_quantity(max_spend: float, price: float, board: AShareBoard, fee_config: dict) -> int:
    if max_spend <= 0 or price <= 0:
        return 0
    commission_rate = float(fee_config["a_share_commission_rate"])
    transfer_rate = float(fee_config["a_share_transfer_fee_rate"])
    min_commission = float(fee_config["a_share_min_commission"])
    rough = int(max_spend / price)
    increment = 1 if board is AShareBoard.STAR else 100
    minimum = 200 if board is AShareBoard.STAR else 100
    quantity = rough - (rough % increment)
    while quantity >= minimum:
        gross = quantity * price
        commission = max(min_commission, gross * commission_rate)
        transfer_fee = gross * transfer_rate
        if gross + commission + transfer_fee <= max_spend:
            return quantity
        quantity -= increment
    return 0


def build_a_share_execution_plan(
    ticker: str,
    curr_date: str,
    rating: str,
    config: dict,
) -> str:
    """Render a deterministic order envelope without inventing account inputs."""
    if not is_a_share_symbol(ticker):
        return f"DATA_UNAVAILABLE: {ticker} is not a supported A-share symbol."
    canonical = normalize_a_share_symbol(ticker)
    normalized_rating = rating.strip().lower()
    if normalized_rating not in _TARGET_WEIGHTS:
        normalized_rating = "hold"
    target_weight = _TARGET_WEIGHTS[normalized_rating]
    try:
        price, price_date = _latest_raw_price(ticker, curr_date)
    except Exception as exc:  # noqa: BLE001 - final decision must survive data-source failure
        return f"DATA_UNAVAILABLE: execution price unavailable for {canonical} ({exc})."

    board = classify_a_share_board(ticker)
    defaults = {
        "a_share_account_cash": 0.0,
        "a_share_available_shares": 0,
        "a_share_commission_rate": 0.0003,
        "a_share_min_commission": 5.0,
        "a_share_transfer_fee_rate": 0.00001,
        "a_share_stamp_duty_rate": 0.0005,
    }
    fees = {key: config.get(key, value) for key, value in defaults.items()}
    cash = max(0.0, float(fees["a_share_account_cash"] or 0.0))
    available_shares = max(0, int(fees["a_share_available_shares"] or 0))
    commission_rate = float(fees["a_share_commission_rate"])
    min_commission = float(fees["a_share_min_commission"])
    transfer_rate = float(fees["a_share_transfer_fee_rate"])
    stamp_rate = float(fees["a_share_stamp_duty_rate"])

    lot_text = (
        "minimum 200 shares, then 1-share increments"
        if board is AShareBoard.STAR
        else "100 shares"
    )
    side = "NONE"
    quantity: int | None = None
    if target_weight is None:
        quantity = 0
    elif normalized_rating in {"buy", "overweight"}:
        if cash > 0:
            total_equity = cash + available_shares * price
            target_value = total_equity * target_weight
            additional_value = max(0.0, target_value - available_shares * price)
            spend_cap = min(cash, additional_value)
            quantity = _buy_quantity(spend_cap, price, board, fees)
            side = "BUY" if quantity > 0 else "NONE"
    elif normalized_rating == "underweight":
        if available_shares > 0:
            total_equity = cash + available_shares * price
            target_shares = math.floor((total_equity * target_weight) / price)
            quantity = max(0, available_shares - target_shares)
            side = "SELL" if quantity > 0 else "NONE"
    elif normalized_rating == "sell":
        if available_shares > 0:
            quantity = available_shares
            side = "SELL"

    lines = [
        f"## Deterministic A-share execution envelope for {canonical}",
        f"- Rating: {rating}",
        f"- Target weight: {'maintain current' if target_weight is None else f'{target_weight:.0%}'}",
        f"- Reference price: {price:.2f} CNY ({price_date}, unadjusted close)",
        f"- Board lot: {lot_text}",
        f"- Configured cash: {cash:.2f} CNY",
        f"- Configured sellable shares: {available_shares}",
    ]
    if quantity is None:
        lines.extend(
            [
                "- Order side: unavailable",
                "- Order quantity: unavailable",
                "- To size an order, configure TRADINGAGENTS_A_SHARE_ACCOUNT_CASH and/or TRADINGAGENTS_A_SHARE_AVAILABLE_SHARES.",
            ]
        )
        return "\n".join(lines)

    gross = quantity * price
    commission = max(min_commission, gross * commission_rate) if quantity > 0 else 0.0
    transfer_fee = gross * transfer_rate if quantity > 0 else 0.0
    stamp_duty = gross * stamp_rate if side == "SELL" else 0.0
    total_fees = commission + transfer_fee + stamp_duty
    lines.extend(
        [
            f"- Order side: {side}",
            f"- Order quantity: {quantity} shares",
            f"- Gross amount: {gross:.2f} CNY",
            f"- Commission: {commission:.2f} CNY",
            f"- Transfer fee: {transfer_fee:.2f} CNY",
            f"- Stamp duty: {stamp_duty:.2f} CNY",
            f"- Estimated total fees: {total_fees:.2f} CNY",
            "- T+1: sell quantity must not exceed shares that are already settled and explicitly configured as sellable.",
            "- This is an execution envelope, not an order submission; verify live price limits, suspension, and broker fees before trading.",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_a_share_execution_plan"]
