import sys

import pandas as pd
import pytest

from tradingagents.agents.utils.a_share_tools import (
    get_a_share_announcements,
    get_a_share_announcement_events,
    get_a_share_dragon_tiger,
    get_a_share_limit_pool,
    get_a_share_relative_comparison,
    get_a_share_shareholder_count,
    get_a_share_trade_status,
    get_a_share_tools_for_analyst,
)


class FakeAkShareSpecialty:
    def stock_lhb_stock_detail_em(self, symbol, start_date, end_date):
        self.dragon_tiger_call = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        }
        return pd.DataFrame(
            [
                {
                    "\u4ee3\u7801": symbol,
                    "\u540d\u79f0": "\u8d35\u5dde\u8305\u53f0",
                    "\u4e0a\u699c\u65e5\u671f": "2026-01-05",
                    "\u4e70\u5165\u989d": 1000,
                }
            ]
        )

    def stock_zt_pool_em(self, date):
        self.limit_pool_call = {"date": date}
        return pd.DataFrame(
            [
                {
                    "\u4ee3\u7801": "600519",
                    "\u540d\u79f0": "\u8d35\u5dde\u8305\u53f0",
                    "\u6da8\u505c\u539f\u56e0": "\u6d88\u8d39",
                }
            ]
        )

    def stock_zh_a_gdhs_detail_em(self, symbol):
        self.shareholder_call = {"symbol": symbol}
        return pd.DataFrame(
            [
                {
                    "\u80a1\u4e1c\u6237\u6570": 120000,
                    "\u622a\u6b62\u65e5\u671f": "2025-12-31",
                }
            ]
        )

    def stock_individual_notice_report(self, security, symbol, begin_date, end_date):
        self.notice_call = {
            "security": security,
            "symbol": symbol,
            "begin_date": begin_date,
            "end_date": end_date,
        }
        return pd.DataFrame(
            [{"代码": security, "公告日期": "2026-01-04", "公告标题": "年度业绩预告"}]
        )


@pytest.fixture()
def fake_akshare(monkeypatch):
    fake = FakeAkShareSpecialty()
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return fake


@pytest.mark.unit
def test_a_share_tool_bundles_only_for_a_shares():
    market_names = [tool.name for tool in get_a_share_tools_for_analyst("market", "600519")]
    news_names = [tool.name for tool in get_a_share_tools_for_analyst("news", "600519.SS")]
    fundamentals_names = [
        tool.name for tool in get_a_share_tools_for_analyst("fundamentals", "300750")
    ]

    assert "get_a_share_limit_pool" in market_names
    assert "get_a_share_dragon_tiger" in news_names
    assert "get_a_share_shareholder_count" in fundamentals_names
    assert get_a_share_tools_for_analyst("market", "AAPL") == []


@pytest.mark.unit
def test_dragon_tiger_tool_formats_fake_akshare(fake_akshare):
    result = get_a_share_dragon_tiger.func("600519", "2026-01-01", "2026-01-10")

    assert fake_akshare.dragon_tiger_call == {
        "symbol": "600519",
        "start_date": "20260101",
        "end_date": "20260110",
    }
    assert "# A-share dragon tiger data for 600519.SS" in result
    assert "\u8d35\u5dde\u8305\u53f0" in result


@pytest.mark.unit
def test_limit_pool_filters_to_ticker(fake_akshare):
    result = get_a_share_limit_pool.func("600519", "2026-01-05")

    assert fake_akshare.limit_pool_call == {"date": "20260105"}
    assert "# A-share limit-up/limit-down pool data for 600519.SS" in result
    assert "\u6da8\u505c\u539f\u56e0" in result


@pytest.mark.unit
def test_shareholder_tool_formats_fake_akshare(fake_akshare):
    result = get_a_share_shareholder_count.func("600519", "2026-01-05")

    assert fake_akshare.shareholder_call == {"symbol": "600519"}
    assert "# A-share shareholder count data for 600519.SS" in result
    assert "\u80a1\u4e1c\u6237\u6570" in result


@pytest.mark.unit
def test_specialty_tool_rejects_non_a_share(fake_akshare):
    result = get_a_share_dragon_tiger.func("AAPL", "2026-01-01", "2026-01-10")

    assert result.startswith("DATA_UNAVAILABLE")
    assert not hasattr(fake_akshare, "dragon_tiger_call")


@pytest.mark.unit
def test_limit_pool_continues_after_first_pool_has_other_ticker(monkeypatch):
    class Pools:
        def stock_zt_pool_em(self, date):
            return pd.DataFrame([{"代码": "000001", "名称": "平安银行"}])

        def stock_zt_pool_dtgc_em(self, date):
            return pd.DataFrame([{"代码": "600519", "名称": "贵州茅台", "状态": "跌停"}])

    monkeypatch.setitem(sys.modules, "akshare", Pools())

    result = get_a_share_limit_pool.func("600519", "2026-01-05")

    assert "贵州茅台" in result
    assert "stock_zt_pool_dtgc_em" in result


@pytest.mark.unit
def test_official_announcements_are_ticker_and_date_filtered(fake_akshare):
    result = get_a_share_announcements.func("600519", "2026-01-01", "2026-01-05")

    assert fake_akshare.notice_call == {
        "security": "600519",
        "symbol": "全部",
        "begin_date": "20260101",
        "end_date": "20260105",
    }
    assert "年度业绩预告" in result
    assert "Scope: ticker" in result


@pytest.mark.unit
def test_market_bundle_includes_deterministic_trade_status():
    names = [tool.name for tool in get_a_share_tools_for_analyst("market", "600519")]
    assert get_a_share_trade_status.name in names
    assert get_a_share_relative_comparison.name in names


@pytest.mark.unit
def test_announcement_events_are_deterministically_classified(fake_akshare):
    result = get_a_share_announcement_events.func("600519", "2026-01-01", "2026-01-05")

    assert "Category | earnings" in result
    assert "Risk | medium" in result
    assert "年度业绩预告" in result


@pytest.mark.unit
def test_relative_comparison_calculates_stock_and_benchmark_returns(monkeypatch):
    from tradingagents.agents.utils import a_share_tools

    stock = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-30"]),
            "Close": [100.0, 110.0],
            "Open": [100.0, 110.0],
            "High": [100.0, 110.0],
            "Low": [100.0, 110.0],
            "Volume": [1000, 1200],
        }
    )
    index = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-30"]),
            "close": [4000.0, 4200.0],
        }
    )
    monkeypatch.setattr(
        a_share_tools,
        "_stock_history_frame",
        lambda *args, **kwargs: ("600519.SS", stock),
    )
    monkeypatch.setattr(
        a_share_tools,
        "_load_a_share_benchmark_frame",
        lambda *args, **kwargs: ("000300.SS", index),
    )
    monkeypatch.setattr(
        a_share_tools,
        "_resolve_industry_snapshot",
        lambda *args, **kwargs: {"industry": "食品制造业", "industry_pe": 25.0, "stock_pe": 30.0},
    )

    result = get_a_share_relative_comparison.func("600519", "2026-01-30", 30)

    assert "Stock return: +10.00%" in result
    assert "Benchmark return: +5.00%" in result
    assert "Excess return: +5.00%" in result
    assert "Industry PE: 25.0" in result


@pytest.mark.unit
def test_industry_snapshot_matches_cninfo_code_and_weighted_pe(monkeypatch):
    from tradingagents.agents.utils.a_share_tools import _resolve_industry_snapshot

    class IndustryData:
        def stock_industry_change_cninfo(self, symbol, start_date, end_date):
            return pd.DataFrame(
                [{"行业大类": "酒、饮料和精制茶制造业", "行业编码": "C15"}]
            )

        def stock_industry_pe_ratio_cninfo(self, symbol, date):
            return pd.DataFrame(
                [
                    {
                        "行业编码": "C15",
                        "行业名称": "酒、饮料和精制茶制造业",
                        "静态市盈率-加权平均": 19.08,
                    }
                ]
            )

    monkeypatch.setitem(sys.modules, "akshare", IndustryData())

    snapshot = _resolve_industry_snapshot("600519", "2025-06-30")

    assert snapshot["industry"] == "酒、饮料和精制茶制造业"
    assert snapshot["industry_pe"] == 19.08
    assert snapshot["valuation_source"].endswith("20250630")


@pytest.mark.unit
def test_trade_status_uses_raw_reference_price_and_board_lot(monkeypatch):
    from tradingagents.agents.utils import a_share_tools

    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "Open": [99.0, 101.0],
            "High": [101.0, 106.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 105.0],
            "Volume": [1000, 1200],
        }
    )
    monkeypatch.setattr(
        a_share_tools,
        "_stock_history_frame",
        lambda *args, **kwargs: ("600519.SS", raw),
    )

    result = get_a_share_trade_status.func("600519", "2026-01-05")

    assert "Daily-limit reference close (unadjusted): 100.0" in result
    assert "Estimated upper/lower limit: 110.00 / 90.00" in result
    assert "Minimum buy lot: 100 shares" in result
