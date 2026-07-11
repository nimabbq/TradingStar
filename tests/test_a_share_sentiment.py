import sys

import pandas as pd
import pytest

from tradingagents.dataflows.a_share_sentiment import (
    _sentiment_phase,
    build_a_share_sentiment_snapshot,
)
from tradingagents.agents.analysts import sentiment_analyst


class FakeSentimentAkShare:
    def stock_zt_pool_em(self, date):
        return pd.DataFrame(
            [{"代码": f"6005{i:02d}", "连板数": 2 if i < 3 else 1} for i in range(12)]
        )

    def stock_zt_pool_dtgc_em(self, date):
        return pd.DataFrame([{"代码": "000001"}, {"代码": "000002"}])

    def stock_zt_pool_zbgc_em(self, date):
        return pd.DataFrame([{"代码": "300001"}])


@pytest.mark.unit
def test_a_share_sentiment_snapshot_has_deterministic_market_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", FakeSentimentAkShare())

    result = build_a_share_sentiment_snapshot("600519", "2026-01-05")

    assert "Limit-up count: 12" in result
    assert "Limit-down count: 2" in result
    assert "Broken-board count: 1" in result
    assert "Maximum streak: 2" in result
    assert "Sentiment phase:" in result


@pytest.mark.unit
def test_a_share_sentiment_rejects_non_a_share():
    result = build_a_share_sentiment_snapshot("AAPL", "2026-01-05")
    assert result.startswith("DATA_UNAVAILABLE")


def test_high_broken_board_rate_takes_precedence_over_expansion():
    assert _sentiment_phase(92, 4, 91) == "high-divergence"


@pytest.mark.unit
def test_a_share_prefetch_does_not_call_us_social_sources(monkeypatch):
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *args: "A-share news")
    monkeypatch.setattr(
        sentiment_analyst,
        "build_a_share_sentiment_snapshot",
        lambda *args: "local breadth",
    )
    monkeypatch.setattr(
        sentiment_analyst,
        "fetch_stocktwits_messages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("StockTwits must not run")),
    )
    monkeypatch.setattr(
        sentiment_analyst,
        "fetch_reddit_posts",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Reddit must not run")),
    )

    sources = sentiment_analyst._prefetch_sentiment_sources(
        "600519", "2026-01-01", "2026-01-05"
    )

    assert sources == {
        "mode": "a_share",
        "news": "A-share news",
        "local_market": "local breadth",
    }
