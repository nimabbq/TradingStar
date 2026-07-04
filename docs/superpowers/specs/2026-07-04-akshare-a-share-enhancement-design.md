# AkShare A-Share Enhancement Design

## Goal

Upgrade TradingAgents from basic Yahoo-style A-share symbol support to an A-share-aware analysis pipeline. A-share runs should prefer AkShare data, understand Shanghai/Shenzhen/Beijing board rules, and expose China-market-specific signals such as limit-up/limit-down pools, northbound capital, margin financing, dragon-tiger lists, and sector fund flow.

## Current Context

TradingAgents currently routes market, technical, fundamental, news, macro, and prediction-market tool calls through `tradingagents.dataflows.interface.route_to_vendor`. The configured data vendors are category-level strings such as `yfinance` or `alpha_vantage`, with optional tool-level overrides.

The existing A-share support is limited to accepting Yahoo Finance suffixes such as `600519.SS` and `000001.SZ`, plus benchmark mapping for `.SS` and `.SZ`. The pipeline does not currently normalize bare six-digit A-share codes, does not prefer domestic A-share data sources, and does not inject A-share trading rules into agent prompts.

## Architecture

The enhancement adds three bounded units:

1. `tradingagents.dataflows.a_share_rules`
   - Identifies A-share symbols and normalizes bare six-digit codes to Yahoo-style suffixed symbols.
   - Classifies boards: Shanghai main board, STAR Market, Shenzhen main board, ChiNext, Beijing Stock Exchange.
   - Produces deterministic rule context for agent prompts.

2. `tradingagents.dataflows.akshare`
   - Provides vendor-compatible functions for A-share market data, fundamentals, financial statements, news, and A-share-specific datasets.
   - Imports AkShare lazily so non-A-share users do not pay import cost and missing AkShare produces a vendor-not-configured error that the existing router can handle.

3. A-share specialty agent tools
   - Adds explicit tools for market-specific signals that do not fit existing generic vendor methods: dragon-tiger list, northbound capital, margin financing, limit-up/limit-down pools, sector fund flow, shareholder count, institutional holdings, lock-up expiry, and dividend/allotment events.

## Data Vendor Behavior

AkShare is registered as vendor key `akshare`.

For A-share symbols, the default category vendor chain becomes:

- `core_stock_apis`: `akshare,yfinance,alpha_vantage`
- `technical_indicators`: `akshare,yfinance,alpha_vantage`
- `fundamental_data`: `akshare,yfinance,alpha_vantage`
- `news_data`: `akshare,yfinance,alpha_vantage`

For non-A-share symbols, the existing default behavior remains unchanged.

The router still respects explicit user configuration. If code programmatically sets `data_vendors` or `tool_vendors`, that explicit configuration remains the configured chain and no hidden fallback is added. Environment-variable control for data vendors is outside this first implementation.

## AkShare Vendor Surface

The first implementation maps AkShare into the existing vendor surface:

- `get_stock_data`
  - Uses `ak.stock_zh_a_hist`.
  - Accepts either `600519`, `600519.SS`, `000001.SZ`, or `.BJ` symbols.
  - Returns a CSV-like string compatible with existing market analyst prompts.
  - Uses date format conversion from `YYYY-MM-DD` to AkShare `YYYYMMDD`.

- `get_indicators`
  - Uses AkShare historical OHLCV as the data frame source.
  - Reuses the existing stockstats indicator names: `close_50_sma`, `close_200_sma`, `close_10_ema`, `macd`, `macds`, `macdh`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `atr`, `vwma`, `mfi`.

- `get_fundamentals`
  - Uses `ak.stock_individual_info_em` and `ak.stock_financial_abstract` when available.
  - Returns name, market, industry, listing date, total share capital, circulating share capital, total market value, circulating market value, latest financial summary fields, and a source timestamp.

- `get_balance_sheet`, `get_cashflow`, `get_income_statement`
  - Uses AkShare financial statement endpoints that return A-share financial reports.
  - Filters records to the analysis date when date columns are present.
  - Returns CSV text with source and report frequency information.

- `get_news`
  - Uses `ak.stock_news_em`.
  - Returns recent stock-specific news with title, source, publication time, and URL when available.

When AkShare returns an empty data frame for a requested symbol, the vendor raises `NoMarketDataError` so the existing router can emit the standard `NO_DATA_AVAILABLE` sentinel.

## A-Share Specialty Tools

Add `tradingagents.agents.utils.a_share_tools` with these LangChain tools:

- `get_a_share_dragon_tiger(symbol, trade_date=None)`
- `get_a_share_northbound_flow(symbol=None, trade_date=None)`
- `get_a_share_margin_financing(symbol, trade_date=None)`
- `get_a_share_limit_pool(trade_date, direction="up")`
- `get_a_share_sector_fund_flow(trade_date=None, sector_type="industry")`
- `get_a_share_shareholder_count(symbol)`
- `get_a_share_institutional_holdings(symbol)`
- `get_a_share_lockup_expiry(symbol)`
- `get_a_share_dividend_allotment(symbol)`

These tools are only advertised to agents for A-share runs. They return explicit `DATA_UNAVAILABLE` text when AkShare is not installed, an endpoint is temporarily unavailable, or the endpoint does not cover the symbol.

The specialty tools are added to:

- Market analyst: limit pools, sector fund flow, margin financing, northbound flow.
- News analyst: dragon-tiger list, northbound flow, sector fund flow.
- Fundamentals analyst: shareholder count, institutional holdings, lock-up expiry, dividend/allotment.

## A-Share Symbol Rules

`a_share_rules` recognizes and normalizes:

- `600`, `601`, `603`, `605` -> Shanghai main board, `.SS`
- `688`, `689` -> STAR Market, `.SS`
- `000`, `001`, `002`, `003` -> Shenzhen main board, `.SZ`
- `300`, `301` -> ChiNext, `.SZ`
- `8`, `4`, `9` six-digit codes -> Beijing Stock Exchange, `.BJ`

Already suffixed `.SS`, `.SZ`, and `.BJ` symbols are preserved after upper-casing.

The module provides:

- `is_a_share_symbol(symbol: str) -> bool`
- `normalize_a_share_symbol(symbol: str) -> str`
- `classify_a_share_board(symbol: str) -> AShareBoard`
- `build_a_share_rule_context(symbol: str, trade_date: str | None = None, identity: Mapping[str, str] | None = None) -> str`

## A-Share Rule Context

For A-share symbols, `build_instrument_context` appends deterministic rule context covering:

- Trading currency is CNY.
- Standard settlement is T+1 for stocks; the model must not assume US-style intraday sell-after-buy.
- Standard board lot is 100 shares; STAR Market minimum buy order is 200 shares, then increments of 1 share where applicable.
- Regular continuous auction has a midday break; analysis must avoid assuming uninterrupted US-style sessions.
- Opening call auction, continuous auction, and closing call auction are relevant to intraday liquidity and gap interpretation.
- Daily price limit:
  - Main boards: generally 10%.
  - Risk-warning ST and `*ST`: generally 5%.
  - STAR Market and ChiNext: generally 20% after the initial no-limit period.
  - Beijing Stock Exchange: generally 30% after the initial no-limit period.
  - New listings may have no daily limit during the first five trading days, depending on board and current exchange rules.
- Limit-up and limit-down states can make a nominal buy/sell recommendation unexecutable.
- Suspensions, ex-rights/ex-dividend adjustments, disclosure events, and regulatory risk warnings can distort price series and should be checked before making strong claims.

The context is phrased as constraints, not as a complete legal manual. The LLM must use retrieved data when available and flag missing data rather than inventing exact values.

## Configuration

Add AkShare as an optional dependency:

```toml
[project.optional-dependencies]
a-share = [
    "akshare>=1.17.0",
]
```

AkShare is not placed in mandatory dependencies in the first implementation, because it is large and primarily useful for China-market runs. If AkShare is absent and the user analyzes A-shares, the vendor chain falls back to the next configured vendor and reports the absence in logs.

Add config key:

```python
"a_share_data_vendors": {
    "core_stock_apis": "akshare,yfinance,alpha_vantage",
    "technical_indicators": "akshare,yfinance,alpha_vantage",
    "fundamental_data": "akshare,yfinance,alpha_vantage",
    "news_data": "akshare,yfinance,alpha_vantage",
}
```

This keeps the existing global `data_vendors` unchanged for non-A-share assets.

## Error Handling

- Missing AkShare raises `VendorNotConfiguredError`.
- Empty AkShare responses raise `NoMarketDataError`.
- Endpoint failures raise normal exceptions, allowing the existing router to log the broken vendor and try the next configured vendor.
- Specialty tools catch vendor exceptions and return `DATA_UNAVAILABLE` because they are enrichment data, not mandatory inputs.
- No code path silently substitutes non-A-share data for A-share data without the router recording the fallback.

## Testing Strategy

Unit tests cover:

- Bare A-share symbol normalization and board classification.
- `.SS`, `.SZ`, `.BJ` preservation.
- A-share rule context contains settlement, lot size, price-limit, and board-specific rules.
- Non-A-share symbols do not receive A-share context.
- Vendor routing accepts `akshare` and uses it when configured.
- AkShare OHLCV formatting using monkeypatched fake AkShare data frames.
- Missing AkShare produces `VendorNotConfiguredError`.
- Specialty tools return formatted text for fake AkShare data and `DATA_UNAVAILABLE` for unavailable endpoints.
- `build_instrument_context` injects A-share context for A-share symbols.

Integration smoke test, optional:

- When AkShare is installed, call one liquid symbol such as `600519` or `000001.SZ` for a short date range and confirm non-empty output. This test is marked integration and skipped when AkShare is missing.

## Non-Goals For The First Implementation

- No guarantee that every AkShare endpoint is stable across all historical dates.
- No automatic official exchange holiday calendar implementation in this pass.
- No brokerage-specific order simulation.
- No real-time trading execution.
- No legal or regulatory compliance certification; the rule context is analysis guidance.

## Acceptance Criteria

- A user can run an A-share analysis with symbols such as `600519`, `600519.SS`, `300750`, `688981`, and `430047.BJ`.
- The analysis prompt includes A-share trading rules for A-share symbols and omits them for US/HK/crypto symbols.
- AkShare is the first attempted data source for A-share market, technical, fundamental, and news data when no explicit override is provided.
- A-share specialty tools are available to the relevant analyst nodes.
- Unit tests pass without AkShare installed by using monkeypatches.
- The project remains usable for non-A-share users without installing AkShare.
