# A-share Sentiment, Execution, and Review Loop

## Contracts

1. A-share sentiment uses dated local market breadth, limit pools, broken-board counts, hot rank, and official/news context. It does not call StockTwits or Reddit. Current-only endpoints are never attached to a historical analysis date.
2. Execution sizing is deterministic and optional. Cash and sellable shares default to zero, meaning no quantity is invented. When configured, prices are unadjusted, buy quantities follow board lots, fees are itemized, and sell plans use only explicitly sellable shares.
3. A-share deferred review uses AkShare prices and 5/20/60 trading-day horizons. A pending entry resolves only after all configured horizons are available; the longest horizon remains the compact log tag while every horizon appears in the reflection.
4. Resolved decisions expose deterministic win-rate and average-return statistics in future same-ticker context.

## Verification

- Unit tests for sentiment phase metrics and A-share source routing.
- Unit tests for lot rounding, fees, no-account behavior, and PM integration.
- Unit tests for multi-horizon completion, incomplete horizons, and performance summary.
- Live AkShare smoke tests and the full repository test suite.
