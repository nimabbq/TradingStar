"""Deterministic local-market sentiment snapshot for mainland A-shares."""

from __future__ import annotations

import pandas as pd

from .a_share_rules import is_a_share_symbol, normalize_a_share_symbol
from .akshare import _akshare, _em_symbol, _yyyymmdd


def _safe_frame(endpoint_name: str, **kwargs) -> tuple[pd.DataFrame, str | None]:
    endpoint = getattr(_akshare(), endpoint_name, None)
    if endpoint is None:
        return pd.DataFrame(), f"{endpoint_name}: unavailable"
    try:
        data = endpoint(**kwargs)
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return frame.copy(), None
    except Exception as exc:  # noqa: BLE001 - one sentiment source must not abort analysis
        return pd.DataFrame(), f"{endpoint_name}: {type(exc).__name__}"


def _maximum_streak(frame: pd.DataFrame) -> int | None:
    for column in ("\u8fde\u677f\u6570", "\u8fde\u677f", "\u8fde\u7eed\u6da8\u505c\u5929\u6570"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        if values.notna().any():
            return int(values.max())
    return None


def _sentiment_phase(limit_up: int, limit_down: int, broken: int) -> str:
    total_signal = limit_up + limit_down
    if total_signal == 0:
        return "unknown"
    break_rate = broken / max(1, limit_up + broken)
    if limit_down >= max(10, limit_up):
        return "retreat"
    if limit_up <= 10 and limit_down >= limit_up / 2:
        return "ice-point"
    if limit_up >= 70 and limit_down <= 10 and break_rate < 0.25:
        return "climax"
    if break_rate >= 0.4:
        return "high-divergence"
    if limit_up >= 35 and limit_up >= limit_down * 3:
        return "expansion"
    return "repair-or-divergence"


def build_a_share_sentiment_snapshot(ticker: str, curr_date: str) -> str:
    """Return dated breadth/limit-pool signals without US social-media proxies."""
    if not is_a_share_symbol(ticker):
        return f"DATA_UNAVAILABLE: {ticker} is not a supported A-share symbol."
    canonical = normalize_a_share_symbol(ticker)
    date = _yyyymmdd(curr_date)
    up, up_error = _safe_frame("stock_zt_pool_em", date=date)
    down, down_error = _safe_frame("stock_zt_pool_dtgc_em", date=date)
    broken, broken_error = _safe_frame("stock_zt_pool_zbgc_em", date=date)

    current_only_lines: list[str] = []
    errors = [error for error in (up_error, down_error, broken_error) if error]
    analysis_date = pd.to_datetime(curr_date, errors="coerce")
    is_today = not pd.isna(analysis_date) and analysis_date.normalize() == pd.Timestamp.today().normalize()
    if is_today:
        activity, error = _safe_frame("stock_market_activity_legu")
        if error:
            errors.append(error)
        elif not activity.empty:
            current_only_lines.append("- Current market activity snapshot:\n" + activity.tail(20).to_csv(index=False))

        hot, error = _safe_frame("stock_hot_rank_latest_em", symbol=_em_symbol(canonical))
        if error:
            errors.append(error)
        elif not hot.empty:
            current_only_lines.append("- Current ticker popularity snapshot:\n" + hot.tail(10).to_csv(index=False))

    up_count, down_count, broken_count = len(up), len(down), len(broken)
    streak = _maximum_streak(up)
    phase = _sentiment_phase(up_count, down_count, broken_count)
    lines = [
        f"## A-share local market sentiment for {canonical}",
        f"- Analysis date: {curr_date}",
        f"- Limit-up count: {up_count}",
        f"- Limit-down count: {down_count}",
        f"- Broken-board count: {broken_count}",
        f"- Maximum streak: {streak if streak is not None else 'unknown'}",
        f"- Broken-board rate: {broken_count / max(1, up_count + broken_count):.1%}",
        f"- Sentiment phase: {phase}",
        "- Phase is a deterministic breadth label, not a directional price forecast.",
    ]
    if current_only_lines:
        lines.extend(current_only_lines)
    if not is_today:
        lines.append("- Current-only popularity/activity endpoints were excluded from this historical snapshot.")
    if errors:
        lines.append("- Unavailable sources: " + "; ".join(errors))
    return "\n".join(lines)


__all__ = ["build_a_share_sentiment_snapshot"]
