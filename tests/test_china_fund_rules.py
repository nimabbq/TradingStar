import pytest

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows.china_fund_rules import (
    build_china_fund_rule_context,
    extract_china_fund_code,
    is_china_fund_symbol,
    normalize_china_fund_symbol,
)
from tradingagents.dataflows.symbol_utils import normalize_symbol


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("005827", "005827.FUND"),
        ("005827.fund", "005827.FUND"),
        ("FUND:005827", "005827.FUND"),
        (" fund:005827 ", "005827.FUND"),
    ],
)
def test_normalizes_china_mutual_fund_symbols(raw, expected):
    assert normalize_china_fund_symbol(raw) == expected
    assert normalize_symbol(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["005827", "005827.FUND", "FUND:005827"])
def test_identifies_china_mutual_fund_symbols(raw):
    assert is_china_fund_symbol(raw) is True
    assert extract_china_fund_code(raw) == "005827"


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["600519", "300750", "510300.SS", "159915.SZ", "AAPL"])
def test_does_not_confuse_stocks_or_exchange_etfs_with_funds(raw):
    assert is_china_fund_symbol(raw) is False


@pytest.mark.unit
def test_fund_rule_context_uses_fund_specific_constraints():
    context = build_china_fund_rule_context("005827", identity={"fund_name": "易方达蓝筹精选混合"})

    assert "005827.FUND" in context
    assert "China mutual fund" in context
    assert "易方达蓝筹精选混合" in context
    assert "NAV" in context
    assert "not an intraday exchange-traded stock" in context
    assert "limit-up" in context


@pytest.mark.unit
def test_instrument_context_includes_fund_guidance():
    context = build_instrument_context("005827.FUND", asset_type="fund")

    assert "005827.FUND" in context
    assert "China mutual fund" in context
    assert "not an intraday exchange-traded stock" in context
    assert "company fundamentals" not in context
