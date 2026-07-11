from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from tradingagents.dataflows.a_share_rules import AShareBoard, classify_a_share_board, is_a_share_symbol
from tradingagents.dataflows.akshare import (
    _akshare,
    _canonical_and_code,
    _format_frame,
    _stock_history_frame,
    _yyyymmdd,
)
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError

_CODE_COLUMNS = (
    "\u4ee3\u7801",
    "\u80a1\u7968\u4ee3\u7801",
    "\u8bc1\u5238\u4ee3\u7801",
    "code",
    "symbol",
)
_DATE_COLUMNS = (
    "\u65e5\u671f",
    "\u4ea4\u6613\u65e5\u671f",
    "\u4e0a\u699c\u65e5\u671f",
    "\u516c\u544a\u65e5\u671f",
    "\u622a\u6b62\u65e5\u671f",
    "\u89e3\u7981\u65e5\u671f",
    "\u53d1\u5e03\u65f6\u95f4",
    "date",
)


def _market_for(canonical: str) -> str:
    suffix = canonical.split(".", 1)[1]
    return {"SS": "sh", "SZ": "sz", "BJ": "bj"}[suffix]


def _as_frame(data) -> pd.DataFrame:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    return frame.copy()


_MARKET_AGGREGATE_ENDPOINTS = {
    "stock_hsgt_hist_em",
    "stock_hsgt_north_net_flow_in_em",
    "stock_sector_fund_flow_rank",
}


def _call_first_available(
    candidates: list[tuple[str, list[dict]]],
    detail: str,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, str]:
    ak = _akshare()
    saw_endpoint = False
    last_error: Exception | None = None

    for endpoint_name, kwargs_options in candidates:
        endpoint = getattr(ak, endpoint_name, None)
        if endpoint is None:
            continue
        saw_endpoint = True
        for kwargs in kwargs_options:
            try:
                frame = _as_frame(endpoint(**kwargs))
            except Exception as exc:  # noqa: BLE001 - try the next documented endpoint shape
                last_error = exc
                continue
            if transform is not None and not frame.empty:
                frame = transform(frame)
            if not frame.empty:
                return frame, endpoint_name

    if not saw_endpoint:
        names = ", ".join(name for name, _ in candidates)
        raise VendorNotConfiguredError(f"akshare has none of the required endpoints: {names}")
    if last_error is not None:
        raise last_error
    raise NoMarketDataError("A-share", detail=detail)


def _filter_by_code(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    for column in _CODE_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.extract(r"(\d{6})", expand=False)
        return frame[values == code]
    return frame


def _filter_by_date_range(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return frame
    start = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end = pd.to_datetime(end_date, errors="coerce") if end_date else None
    if end is not None and not pd.isna(end):
        end = end + pd.Timedelta(days=1)

    for column in _DATE_COLUMNS:
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce")
        mask = pd.Series(True, index=frame.index)
        if start is not None and not pd.isna(start):
            mask = mask & (dates.isna() | (dates >= start))
        if end is not None and not pd.isna(end):
            mask = mask & (dates.isna() | (dates < end))
        return frame[mask]
    return frame


def _run_specialty_tool(
    ticker: str,
    title: str,
    candidates: list[tuple[str, list[dict]]],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    filter_code: bool = True,
) -> str:
    if not is_a_share_symbol(ticker):
        return f"DATA_UNAVAILABLE: {ticker} is not a supported A-share symbol."
    try:
        canonical, code = _canonical_and_code(ticker)
        def prepare(candidate: pd.DataFrame) -> pd.DataFrame:
            if filter_code:
                candidate = _filter_by_code(candidate, code)
            return _filter_by_date_range(candidate, start_date, end_date)

        frame, endpoint_name = _call_first_available(
            candidates,
            f"no {title} data",
            transform=prepare,
        )
        scope = "market aggregate context" if endpoint_name in _MARKET_AGGREGATE_ENDPOINTS else "ticker"
        return _format_frame(
            f"# A-share {title} data for {canonical}\n"
            f"# Source endpoint: {endpoint_name}\n# Scope: {scope}",
            frame,
        )
    except VendorNotConfiguredError as exc:
        return f"DATA_UNAVAILABLE: {exc}"
    except NoMarketDataError as exc:
        return f"NO_DATA_AVAILABLE: {exc}"
    except Exception as exc:  # noqa: BLE001 - optional enrichment should not crash the graph
        return f"DATA_UNAVAILABLE: could not retrieve A-share {title} data ({exc})."


@tool
def get_a_share_dragon_tiger(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share dragon-tiger list data for abnormal turnover and active seats."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    return _run_specialty_tool(
        ticker,
        "dragon tiger",
        [
            (
                "stock_lhb_stock_detail_em",
                [
                    {"symbol": code, "start_date": _yyyymmdd(start_date), "end_date": _yyyymmdd(end_date)},
                    {"symbol": code},
                ],
            ),
            (
                "stock_lhb_detail_em",
                [
                    {"start_date": _yyyymmdd(start_date), "end_date": _yyyymmdd(end_date)},
                    {"date": _yyyymmdd(end_date)},
                ],
            ),
            ("stock_lhb_stock_statistic_em", [{"symbol": code}, {"symbol": canonical}]),
        ],
        start_date=start_date,
        end_date=end_date,
    )


@tool
def get_a_share_announcements(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve official A-share disclosures for point-in-time event analysis."""
    code = _canonical_and_code(ticker)[1] if is_a_share_symbol(ticker) else ticker
    return _run_specialty_tool(
        ticker,
        "official announcements",
        [
            (
                "stock_individual_notice_report",
                [
                    {
                        "security": code,
                        "symbol": "\u5168\u90e8",
                        "begin_date": _yyyymmdd(start_date),
                        "end_date": _yyyymmdd(end_date),
                    }
                ],
            ),
            (
                "stock_zh_a_disclosure_report_cninfo",
                [
                    {
                        "symbol": code,
                        "market": "\u6caa\u6df1\u4eac",
                        "keyword": "",
                        "category": "",
                        "start_date": _yyyymmdd(start_date),
                        "end_date": _yyyymmdd(end_date),
                    }
                ],
            ),
        ],
        start_date=start_date,
        end_date=end_date,
    )


@tool
def get_a_share_northbound_flow(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share northbound/Stock Connect flow or holding data."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    return _run_specialty_tool(
        ticker,
        "northbound flow",
        [
            ("stock_hsgt_individual_em", [{"symbol": code}, {"symbol": canonical}]),
            ("stock_hsgt_stock_statistics_em", [{"symbol": code}, {"symbol": canonical}]),
            ("stock_hsgt_hist_em", [{"symbol": "\u6caa\u80a1\u901a"}, {"symbol": "\u6df1\u80a1\u901a"}]),
            ("stock_hsgt_north_net_flow_in_em", [{"symbol": "\u5317\u4e0a"}]),
        ],
        start_date=start_date,
        end_date=end_date,
        filter_code=False,
    )


@tool
def get_a_share_margin_financing(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share margin financing and securities lending data."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    market = _market_for(canonical) if is_a_share_symbol(ticker) else "sh"
    return _run_specialty_tool(
        ticker,
        "margin financing",
        [
            (
                "stock_margin_detail_em",
                [
                    {"symbol": code, "start_date": _yyyymmdd(start_date), "end_date": _yyyymmdd(end_date)},
                    {"symbol": code},
                ],
            ),
            ("stock_margin_detail_sse", [{"date": _yyyymmdd(end_date)}, {"symbol": code}]),
            ("stock_margin_detail_szse", [{"date": _yyyymmdd(end_date)}, {"symbol": code}]),
            ("stock_margin_szse", [{"date": _yyyymmdd(end_date)}]),
            ("stock_margin_underlying_info_szse", [{"date": _yyyymmdd(end_date)}, {"market": market}]),
        ],
        start_date=start_date,
        end_date=end_date,
    )


@tool
def get_a_share_limit_pool(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share limit-up/limit-down pool data around the analysis date."""
    code = _canonical_and_code(ticker)[1] if is_a_share_symbol(ticker) else ticker
    return _run_specialty_tool(
        ticker,
        "limit-up/limit-down pool",
        [
            ("stock_zt_pool_em", [{"date": _yyyymmdd(curr_date)}]),
            ("stock_zt_pool_dtgc_em", [{"date": _yyyymmdd(curr_date)}]),
            ("stock_zt_pool_strong_em", [{"date": _yyyymmdd(curr_date)}]),
            ("stock_zt_pool_previous_em", [{"date": _yyyymmdd(curr_date)}]),
            ("stock_zt_pool_sub_new_em", [{"date": _yyyymmdd(curr_date)}]),
        ],
        start_date=curr_date,
        end_date=curr_date,
        filter_code=bool(code),
    )


@tool
def get_a_share_sector_fund_flow(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share individual or sector fund-flow data."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    market = _market_for(canonical) if is_a_share_symbol(ticker) else "sh"
    return _run_specialty_tool(
        ticker,
        "sector fund flow",
        [
            ("stock_individual_fund_flow", [{"stock": code, "market": market}, {"symbol": code}]),
            (
                "stock_sector_fund_flow_rank",
                [
                    {
                        "indicator": "\u4eca\u65e5",
                        "sector_type": "\u884c\u4e1a\u8d44\u91d1\u6d41",
                    },
                    {},
                ],
            ),
        ],
        start_date=curr_date,
        end_date=curr_date,
        filter_code=False,
    )


@tool
def get_a_share_shareholder_count(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share shareholder-count trend data."""
    code = _canonical_and_code(ticker)[1] if is_a_share_symbol(ticker) else ticker
    return _run_specialty_tool(
        ticker,
        "shareholder count",
        [
            ("stock_zh_a_gdhs_detail_em", [{"symbol": code}]),
            ("stock_hold_num_cninfo", [{"date": _yyyymmdd(curr_date)}]),
        ],
        end_date=curr_date,
        filter_code=True,
    )


@tool
def get_a_share_institutional_holdings(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share institutional holding data when AkShare exposes it."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    return _run_specialty_tool(
        ticker,
        "institutional holdings",
        [
            ("stock_institute_hold", [{"symbol": code}, {"symbol": canonical}]),
            ("stock_institute_hold_detail", [{"stock": code}, {"symbol": code}]),
            ("stock_report_fund_hold", [{"symbol": code}, {"symbol": canonical}]),
        ],
        end_date=curr_date,
    )


@tool
def get_a_share_lockup_expiry(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share restricted-share unlock or lockup-expiry data."""
    canonical, code = _canonical_and_code(ticker) if is_a_share_symbol(ticker) else (ticker, ticker)
    return _run_specialty_tool(
        ticker,
        "lockup expiry",
        [
            ("stock_restricted_release_detail_em", [{"symbol": code}, {"symbol": canonical}]),
            ("stock_restricted_release_queue_em", [{"symbol": code}, {}]),
            ("stock_restricted_release_summary_em", [{"symbol": code}, {}]),
        ],
        start_date=curr_date,
    )


@tool
def get_a_share_dividend_allotment(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """Retrieve A-share dividend, allotment, and distribution history."""
    code = _canonical_and_code(ticker)[1] if is_a_share_symbol(ticker) else ticker
    return _run_specialty_tool(
        ticker,
        "dividend and allotment",
        [
            (
                "stock_history_dividend_detail",
                [
                    {"symbol": code, "indicator": "\u5206\u7ea2"},
                    {"symbol": code, "indicator": "\u914d\u80a1"},
                    {"symbol": code},
                ],
            ),
            ("stock_fhps_detail_em", [{"symbol": code}]),
            ("stock_dividend_cninfo", [{"symbol": code}]),
        ],
        end_date=curr_date,
        filter_code=False,
    )


def _profile_values(frame: pd.DataFrame) -> dict[str, str]:
    for key_column, value_column in (
        ("item", "value"),
        ("\u9879\u76ee", "\u503c"),
    ):
        if key_column in frame.columns and value_column in frame.columns:
            return {
                str(row[key_column]).strip(): str(row[value_column]).strip()
                for _, row in frame.iterrows()
            }
    return {}


def _round_price(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@tool
def get_a_share_trade_status(
    ticker: Annotated[str, "A-share ticker symbol, e.g. 600519"],
    curr_date: Annotated[str, "Analysis date in yyyy-mm-dd format"],
) -> str:
    """Build a dated A-share executability snapshot from raw prices and status lists."""
    if not is_a_share_symbol(ticker):
        return f"DATA_UNAVAILABLE: {ticker} is not a supported A-share symbol."
    canonical, code = _canonical_and_code(ticker)
    board = classify_a_share_board(ticker)
    ak = _akshare()
    analysis_date = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(analysis_date):
        return f"DATA_UNAVAILABLE: invalid analysis date {curr_date!r}."

    try:
        _, history = _stock_history_frame(
            ticker,
            (analysis_date - timedelta(days=45)).strftime("%Y-%m-%d"),
            curr_date,
            adjust="",
        )
        history = history.copy()
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history.dropna(subset=["Date", "Close"]).sort_values("Date")
    except Exception as exc:  # noqa: BLE001
        return f"DATA_UNAVAILABLE: could not build raw-price trade status for {canonical} ({exc})."
    if history.empty:
        return f"NO_DATA_AVAILABLE: no raw price rows for {canonical} on or before {curr_date}."

    latest = history.iloc[-1]
    latest_date = latest["Date"].strftime("%Y-%m-%d")
    if latest["Date"].normalize() == analysis_date.normalize() and len(history) > 1:
        limit_reference = float(history.iloc[-2]["Close"])
    else:
        limit_reference = float(latest["Close"])

    profile = {}
    endpoint = getattr(ak, "stock_individual_info_em", None)
    if endpoint is not None:
        try:
            profile = _profile_values(_as_frame(endpoint(symbol=code)))
        except Exception:
            profile = {}
    name = profile.get("\u80a1\u7968\u7b80\u79f0") or profile.get("\u7b80\u79f0") or "unknown"
    listing_raw = profile.get("\u4e0a\u5e02\u65f6\u95f4") or profile.get("\u4e0a\u5e02\u65e5\u671f")
    listing_date = pd.to_datetime(listing_raw, errors="coerce")

    st_status = "unknown"
    st_endpoint = getattr(ak, "stock_zh_a_st_em", None)
    if st_endpoint is not None and analysis_date.normalize() == pd.Timestamp.today().normalize():
        try:
            st_status = "yes" if not _filter_by_code(_as_frame(st_endpoint()), code).empty else "no"
        except Exception:
            pass

    suspension = "unknown"
    suspension_endpoint = getattr(ak, "stock_tfp_em", None)
    if suspension_endpoint is not None:
        try:
            suspension = (
                "yes"
                if not _filter_by_code(_as_frame(suspension_endpoint(date=_yyyymmdd(curr_date))), code).empty
                else "no"
            )
        except Exception:
            pass

    first_five = "unknown"
    calendar_endpoint = getattr(ak, "tool_trade_date_hist_sina", None)
    if calendar_endpoint is not None and not pd.isna(listing_date):
        try:
            calendar = _as_frame(calendar_endpoint())
            date_column = next((c for c in ("trade_date", "\u4ea4\u6613\u65e5\u671f", "date") if c in calendar.columns), None)
            if date_column:
                dates = pd.to_datetime(calendar[date_column], errors="coerce")
                count = int(((dates >= listing_date) & (dates <= analysis_date)).sum())
                first_five = "yes" if count <= 5 else "no"
        except Exception:
            pass

    lot_size = 200 if board is AShareBoard.STAR else 100
    limit_pct = 0.05 if st_status == "yes" else {
        AShareBoard.SH_MAIN: 0.10,
        AShareBoard.SZ_MAIN: 0.10,
        AShareBoard.STAR: 0.20,
        AShareBoard.CHINEXT: 0.20,
        AShareBoard.BSE: 0.30,
    }[board]
    if first_five == "yes":
        upper = lower = "not applicable during initial no-limit window"
    else:
        upper = _round_price(limit_reference * (1 + limit_pct))
        lower = _round_price(limit_reference * (1 - limit_pct))

    listing_text = "unknown" if pd.isna(listing_date) else listing_date.strftime("%Y-%m-%d")
    return "\n".join(
        [
            f"## A-share executable trade status for {canonical}",
            f"- Analysis date: {curr_date}",
            f"- Name: {name}",
            f"- Board: {board.value}",
            f"- Latest raw-price row: {latest_date}; close={latest['Close']}",
            f"- Daily-limit reference close (unadjusted): {limit_reference}",
            f"- Estimated upper/lower limit: {upper} / {lower}",
            f"- ST status: {st_status}",
            f"- Suspended on analysis date: {suspension}",
            f"- Listing date: {listing_text}; within first five trading days: {first_five}",
            f"- Minimum buy lot: {lot_size} shares",
            "- Stock settlement constraint: T+1; shares bought today cannot be sold today.",
            "- Status fields marked unknown must not be guessed. Limit prices are estimates until verified against the exchange feed.",
        ]
    )


MARKET_A_SHARE_TOOLS = [
    get_a_share_trade_status,
    get_a_share_limit_pool,
    get_a_share_sector_fund_flow,
    get_a_share_margin_financing,
]
NEWS_A_SHARE_TOOLS = [
    get_a_share_announcements,
    get_a_share_dragon_tiger,
    get_a_share_northbound_flow,
    get_a_share_margin_financing,
]
FUNDAMENTALS_A_SHARE_TOOLS = [
    get_a_share_shareholder_count,
    get_a_share_institutional_holdings,
    get_a_share_lockup_expiry,
    get_a_share_dividend_allotment,
]

_A_SHARE_TOOL_BUNDLES = {
    "market": MARKET_A_SHARE_TOOLS,
    "news": NEWS_A_SHARE_TOOLS,
    "fundamentals": FUNDAMENTALS_A_SHARE_TOOLS,
}


def get_a_share_tools_for_analyst(analyst: str, ticker: str):
    if not is_a_share_symbol(ticker):
        return []
    return list(_A_SHARE_TOOL_BUNDLES.get(analyst, []))


ALL_A_SHARE_TOOLS = [
    get_a_share_trade_status,
    get_a_share_announcements,
    get_a_share_dragon_tiger,
    get_a_share_northbound_flow,
    get_a_share_margin_financing,
    get_a_share_limit_pool,
    get_a_share_sector_fund_flow,
    get_a_share_shareholder_count,
    get_a_share_institutional_holdings,
    get_a_share_lockup_expiry,
    get_a_share_dividend_allotment,
]

__all__ = [
    "ALL_A_SHARE_TOOLS",
    "MARKET_A_SHARE_TOOLS",
    "NEWS_A_SHARE_TOOLS",
    "FUNDAMENTALS_A_SHARE_TOOLS",
    "get_a_share_tools_for_analyst",
    "get_a_share_dragon_tiger",
    "get_a_share_announcements",
    "get_a_share_trade_status",
    "get_a_share_northbound_flow",
    "get_a_share_margin_financing",
    "get_a_share_limit_pool",
    "get_a_share_sector_fund_flow",
    "get_a_share_shareholder_count",
    "get_a_share_institutional_holdings",
    "get_a_share_lockup_expiry",
    "get_a_share_dividend_allotment",
]
