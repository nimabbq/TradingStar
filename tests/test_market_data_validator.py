"""Tests for the deterministic market-data verification snapshot (#830/#881)."""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.market_data_validator as validator
import tradingagents.dataflows.stockstats_utils as stockstats_utils


def _sample_ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-05-20")
    closes = [100 + i for i in range(len(dates))]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i for i in range(len(dates))],
    })


@pytest.mark.unit
class TestVerifiedSnapshot:
    def test_excludes_future_rows(self, monkeypatch):
        data = pd.concat([
            _sample_ohlcv(),
            pd.DataFrame({"Date": [pd.Timestamp("2026-06-01")], "Open": [999.0],
                          "High": [999.0], "Low": [999.0], "Close": [999.0], "Volume": [999]}),
        ], ignore_index=True)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)

        snap = validator.build_verified_market_snapshot("COF", "2026-05-13")
        assert "Verified market data snapshot for COF" in snap
        assert "Requested analysis date: 2026-05-13" in snap
        assert "Latest trading row used: 2026-05-13" in snap
        assert "999.00" not in snap          # future row excluded
        assert "boll_lb" in snap             # indicators present

    def test_uses_previous_trading_day_when_date_is_weekend(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        # 2026-05-16 is a Saturday; latest row should be Fri 2026-05-15
        snap = validator.build_verified_market_snapshot("COF", "2026-05-16")
        assert "Latest trading row used: 2026-05-15" in snap
        assert "Recent verified closes" in snap

    def test_raises_when_no_rows_on_or_before_date(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2020-01-01")

    def test_raises_on_empty_data(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2026-05-13")

    def test_look_back_window_capped_at_30(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        snap = validator.build_verified_market_snapshot("COF", "2026-05-20", look_back_days=999)
        # last-N closes table has at most 30 data rows
        close_rows = [ln for ln in snap.splitlines() if ln.startswith("| 2026-")]
        assert 0 < len(close_rows) <= 30

    def test_china_fund_snapshot_uses_akshare_nav_not_yahoo(self, monkeypatch, tmp_path):
        class FakeAkShare:
            def fund_open_fund_info_em(self, symbol, indicator, period):
                self.nav_call = {"symbol": symbol, "indicator": indicator, "period": period}
                return pd.DataFrame(
                    [
                        {
                            "\u51c0\u503c\u65e5\u671f": "2026-06-25",
                            "\u5355\u4f4d\u51c0\u503c": 1.234,
                            "\u65e5\u589e\u957f\u7387": 0.12,
                        },
                        {
                            "\u51c0\u503c\u65e5\u671f": "2026-06-30",
                            "\u5355\u4f4d\u51c0\u503c": 1.256,
                            "\u65e5\u589e\u957f\u7387": 1.78,
                        },
                    ]
                )

        fake = FakeAkShare()

        def fail_yahoo(*args, **kwargs):
            raise AssertionError("fund snapshots must not call Yahoo Finance")

        monkeypatch.setitem(__import__("sys").modules, "akshare", fake)
        monkeypatch.setattr(stockstats_utils.yf, "download", fail_yahoo)
        monkeypatch.setattr(
            stockstats_utils,
            "get_config",
            lambda: {"data_cache_dir": str(tmp_path)},
        )

        snap = validator.build_verified_market_snapshot("005064.FUND", "2026-07-04")

        assert fake.nav_call == {
            "symbol": "005064",
            "indicator": "\u5355\u4f4d\u51c0\u503c\u8d70\u52bf",
            "period": "\u5168\u90e8",
        }
        assert "Verified market data snapshot for 005064.FUND" in snap
        assert "Latest trading row used: 2026-06-30" in snap
        assert "| Close | 1.26 |" in snap

    def test_a_share_snapshot_uses_qfq_akshare_not_yahoo(self, monkeypatch):
        from tradingagents.dataflows import akshare as avendor

        calls = []
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-07-02", "2026-07-03"]),
                "Open": [10.0, 10.2],
                "High": [10.3, 10.5],
                "Low": [9.9, 10.1],
                "Close": [10.2, 10.4],
                "Volume": [1000, 1200],
            }
        )

        def fake_ohlcv(symbol, start_date, end_date, *, adjust):
            calls.append((symbol, adjust))
            return frame

        monkeypatch.setattr(avendor, "get_ohlcv_frame", fake_ohlcv)
        monkeypatch.setattr(
            stockstats_utils.yf,
            "download",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Yahoo must not be used")),
        )

        snap = validator.build_verified_market_snapshot("600519", "2026-07-03")

        assert calls == [("600519.SS", "qfq")]
        assert "Price basis: forward-adjusted (qfq)" in snap


@pytest.mark.unit
class TestTool:
    def test_tool_delegates_to_builder(self, monkeypatch):
        from tradingagents.agents.utils.market_data_validation_tools import (
            get_verified_market_snapshot,
        )
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        out = get_verified_market_snapshot.invoke(
            {"symbol": "COF", "curr_date": "2026-05-20"}
        )
        assert "Verified market data snapshot for COF" in out
