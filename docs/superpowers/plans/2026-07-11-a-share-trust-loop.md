# A-share Data Trust Loop

## Goal

Make mainland China analysis point-in-time safe, price-basis explicit, executable under daily trading constraints, and independently support exchange-traded ETFs.

## Data contracts

1. Financial rows are visible only when their announcement/publication date is on or before the analysis date. Fiscal period is never treated as publication time when a publication column exists.
2. Raw prices (`adjust=""`) are used for executable price levels. Forward-adjusted prices (`adjust="qfq"`) are used for continuous technical indicators and return comparisons. Every rendered price block names its basis.
3. A-share specialty data distinguishes ticker-level observations from market aggregates. Endpoint fallback continues after ticker/date filtering, not merely after a non-empty HTTP response.
4. Trading status is a dated deterministic snapshot covering listing age, ST status, suspension, board rules, lot size, and estimated daily limits. Unknown fields remain explicitly unknown.
5. Mainland exchange ETFs have their own symbol classifier, AkShare vendor, trading-rule context, and routing chain. They are not treated as listed companies or off-exchange mutual funds.

## Delivery order

1. Point-in-time filters and official announcements.
2. Dual price basis and verified snapshot labels.
3. Dynamic trading-status tool.
4. Specialty-tool filtered fallback and aggregate labelling.
5. China ETF provider and routing.

## Verification

Each behavior starts with a regression test. Run focused A-share/ETF tests after each slice and the full pytest suite before commit and push.
