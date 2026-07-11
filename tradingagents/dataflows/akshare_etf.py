"""AkShare vendor for mainland China exchange-traded funds."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .akshare import _OHLCV_COLUMNS, _SUPPORTED_INDICATORS, _akshare, _format_frame, _non_empty_frame
from .china_etf_rules import extract_china_etf_code, normalize_china_etf_symbol
from .errors import NoMarketDataError


def _etf_history_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "",
) -> tuple[str, pd.DataFrame]:
    canonical = normalize_china_etf_symbol(symbol)
    code = extract_china_etf_code(symbol)
    ak = _akshare()
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    try:
        frame = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=requested_start.strftime("%Y%m%d"),
            end_date=requested_end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        price_basis = "forward-adjusted (qfq)" if adjust == "qfq" else "unadjusted"
    except Exception as eastmoney_error:  # noqa: BLE001 - current networks often block EM
        endpoint = getattr(ak, "fund_etf_hist_sina", None)
        if endpoint is None:
            raise eastmoney_error
        market = "sh" if canonical.endswith(".SS") else "sz"
        try:
            frame = endpoint(symbol=f"{market}{code}")
        except Exception:
            raise eastmoney_error
        price_basis = "unadjusted (Sina fallback; qfq unavailable from this source)"
    frame = _non_empty_frame(frame, symbol, canonical, "no ETF OHLCV rows")
    frame = frame.rename(columns=_OHLCV_COLUMNS)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise NoMarketDataError(symbol, canonical, "AkShare returned no ETF date/close columns")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame[(frame["Date"] >= requested_start) & (frame["Date"] <= requested_end)]
    for column in ("Open", "High", "Low", "Close", "Volume", "Amount", "Turnover"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Close"]).sort_values("Date")
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, "no usable ETF OHLCV rows")
    frame.attrs["price_basis"] = price_basis
    return canonical, frame


def get_ohlcv_frame(symbol: str, start_date: str, end_date: str, *, adjust: str = "qfq") -> pd.DataFrame:
    _, frame = _etf_history_frame(symbol, start_date, end_date, adjust=adjust)
    for column in ("Open", "High", "Low", "Volume"):
        if column not in frame.columns:
            frame[column] = 0.0 if column == "Volume" else frame["Close"]
    out = frame[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out.attrs["price_basis"] = frame.attrs.get("price_basis", "unknown")
    return out


def get_stock(
    symbol: Annotated[str, "China ETF code, e.g. 510300 or 159915.SZ"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    canonical, frame = _etf_history_frame(symbol, start_date, end_date, adjust="")
    out = frame.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    basis = frame.attrs.get("price_basis", "unadjusted")
    return _format_frame(
        f"# AkShare ETF stock data for {canonical} from {start_date} to {end_date}\n"
        f"# Price basis: {basis} (execution reference)",
        out,
    )


def get_indicator(
    symbol: Annotated[str, "China ETF code"],
    indicator: Annotated[str, "technical indicator"],
    curr_date: Annotated[str, "Analysis date in YYYY-mm-dd"],
    look_back_days: Annotated[int, "lookback days"],
) -> str:
    if indicator not in _SUPPORTED_INDICATORS:
        raise ValueError(f"Indicator {indicator} is not supported")
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=max(int(look_back_days) + 260, 365))
    canonical, frame = _etf_history_frame(
        symbol,
        start_dt.strftime("%Y-%m-%d"),
        curr_date,
        adjust="qfq",
    )
    stats = frame[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = wrap(stats.rename(columns=str.lower).rename(columns={"date": "Date"}))
    df[indicator]
    cutoff = end_dt - relativedelta(days=look_back_days)
    rows = df[df["Date"] >= cutoff]
    rendered = "\n".join(
        f"{row['Date'].strftime('%Y-%m-%d')}: {row[indicator]}" for _, row in rows.iterrows()
    )
    basis = frame.attrs.get("price_basis", "forward-adjusted (qfq)")
    return (
        f"## AkShare ETF {indicator} values for {canonical}\n\n{rendered}\n\n"
        f"Price basis: {basis}."
    )


def get_fundamentals(
    ticker: Annotated[str, "China ETF code"],
    curr_date: Annotated[str, "Analysis date"] = None,
) -> str:
    canonical = normalize_china_etf_symbol(ticker)
    code = extract_china_etf_code(ticker)
    try:
        spot = _non_empty_frame(_akshare().fund_etf_spot_em(), ticker, canonical, "no ETF snapshot")
    except Exception:
        _, history = _etf_history_frame(
            ticker,
            (pd.Timestamp(curr_date or pd.Timestamp.today()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            pd.Timestamp(curr_date or pd.Timestamp.today()).strftime("%Y-%m-%d"),
            adjust="",
        )
        spot = history.tail(1).copy()
        spot.insert(0, "Code", code)
        return _format_frame(
            f"# AkShare ETF limited profile for {canonical}\n"
            "# Scope: latest exchange row; Eastmoney profile unavailable",
            spot,
        )
    code_column = next((c for c in ("\u4ee3\u7801", "\u57fa\u91d1\u4ee3\u7801", "code") if c in spot.columns), None)
    if code_column:
        spot = spot[spot[code_column].astype(str).str.extract(r"(\d{6})", expand=False) == code]
    if spot.empty:
        raise NoMarketDataError(ticker, canonical, "ETF absent from AkShare spot snapshot")
    return _format_frame(f"# AkShare ETF profile and live metrics for {canonical}", spot)


def get_statement_not_applicable(ticker: str, *args, **kwargs) -> str:
    canonical = normalize_china_etf_symbol(ticker)
    return (
        f"DATA_UNAVAILABLE: {canonical} is an exchange-traded fund; company balance sheet, "
        "cash-flow, and income-statement tools are not applicable."
    )
