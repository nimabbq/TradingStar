import sys

import pandas as pd
import pytest

from tradingagents.dataflows import interface


class FakeAkShareFund:
    def fund_open_fund_info_em(self, symbol, indicator, period="全部"):
        self.nav_call = {"symbol": symbol, "indicator": indicator, "period": period}
        return pd.DataFrame(
            [
                {"净值日期": "2026-01-02", "单位净值": 1.1, "日增长率": 0.2},
                {"净值日期": "2026-01-05", "单位净值": 1.12, "日增长率": 1.82},
                {"净值日期": "2026-01-06", "单位净值": 1.11, "日增长率": -0.89},
            ]
        )

    def fund_individual_basic_info_xq(self, symbol):
        self.basic_call = {"symbol": symbol}
        return pd.DataFrame(
            [
                {"item": "基金代码", "value": symbol},
                {"item": "基金名称", "value": "易方达蓝筹精选混合"},
                {"item": "基金类型", "value": "混合型"},
                {"item": "基金经理", "value": "张坤"},
            ]
        )

    def fund_individual_detail_info_xq(self, symbol):
        self.fee_call = {"symbol": symbol}
        return pd.DataFrame(
            [
                {"费用类型": "买入规则", "条件或名称": "0万<买入金额<100万", "费用": 1.5},
                {"费用类型": "卖出规则", "条件或名称": "持有天数<7天", "费用": 1.5},
            ]
        )

    def fund_portfolio_hold_em(self, symbol, date):
        self.stock_hold_call = {"symbol": symbol, "date": date}
        return pd.DataFrame(
            [
                {
                    "股票代码": "00700",
                    "股票名称": "腾讯控股",
                    "占净值比例": 8.5,
                    "季度": f"{date}年4季度股票投资明细",
                }
            ]
        )

    def fund_portfolio_bond_hold_em(self, symbol, date):
        return pd.DataFrame(
            [
                {
                    "债券代码": "240002",
                    "债券名称": "24附息国债02",
                    "占净值比例": 0.65,
                    "季度": f"{date}年4季度债券投资明细",
                }
            ]
        )

    def fund_portfolio_industry_allocation_em(self, symbol, date):
        return pd.DataFrame(
            [
                {"行业类别": "非必需消费品", "占净值比例": 26.73, "截止时间": f"{date}-12-31"}
            ]
        )


@pytest.fixture()
def fake_akshare_fund(monkeypatch):
    fake = FakeAkShareFund()
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return fake


@pytest.mark.unit
def test_akshare_fund_registered_as_vendor():
    assert "akshare_fund" in interface.VENDOR_LIST
    assert "akshare_fund" in interface.VENDOR_METHODS["get_stock_data"]
    assert "akshare_fund" in interface.VENDOR_METHODS["get_fundamentals"]


@pytest.mark.unit
def test_fund_default_chain_prefers_akshare_fund(fake_akshare_fund):
    result = interface.route_to_vendor("get_stock_data", "005827", "2026-01-01", "2026-01-06")

    assert fake_akshare_fund.nav_call == {
        "symbol": "005827",
        "indicator": "单位净值走势",
        "period": "全部",
    }
    assert "# AkShare fund NAV data for 005827.FUND" in result
    assert "2026-01-05" in result
    assert "1.12" in result


@pytest.mark.unit
def test_explicit_fund_prefix_routes_to_akshare_fund(fake_akshare_fund):
    result = interface.route_to_vendor("get_stock_data", "FUND:005827", "2026-01-01", "2026-01-06")

    assert "# AkShare fund NAV data for 005827.FUND" in result


@pytest.mark.unit
def test_fund_indicators_use_nav_history(fake_akshare_fund):
    result = interface.route_to_vendor("get_indicators", "005827", "rsi", "2026-01-06", 3)

    assert "## AkShare fund rsi values for 005827.FUND" in result
    assert "2026-01-06" in result


@pytest.mark.unit
def test_fundamentals_include_profile_fees_and_holdings(fake_akshare_fund):
    result = interface.route_to_vendor("get_fundamentals", "005827", "2026-01-06")

    assert "# AkShare fund fundamentals for 005827.FUND" in result
    assert "易方达蓝筹精选混合" in result
    assert "张坤" in result
    assert "买入规则" in result
    assert "腾讯控股" in result
    assert "24附息国债02" in result
    assert "非必需消费品" in result


@pytest.mark.unit
def test_fund_statement_tools_return_not_applicable(fake_akshare_fund):
    result = interface.route_to_vendor("get_balance_sheet", "005827", "quarterly", "2026-01-06")

    assert result.startswith("DATA_UNAVAILABLE")
    assert "mutual fund" in result


@pytest.mark.unit
def test_non_fund_default_chain_keeps_yfinance(monkeypatch):
    calls = []

    def yfinance(symbol, *args):
        calls.append(symbol)
        return "YF"

    def akshare_fund(symbol, *args):
        raise AssertionError("fund vendor should not be called for a stock")

    with monkeypatch.context() as m:
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "yfinance", yfinance)
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "akshare_fund", akshare_fund)
        result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

    assert result == "YF"
    assert calls == ["AAPL"]
