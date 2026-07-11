# A-share Decision Context Upgrade

## Scope

Add three deterministic A-share context layers: China macro series, industry/index relative comparison, and structured official-announcement events.

## Contracts

1. China macro data uses explicit indicator aliases and dated AkShare series. Rows after the analysis date are excluded. Failed endpoints continue to documented fallbacks and optional macro failure never aborts the graph.
2. Relative comparison uses forward-adjusted stock prices, a dated broad-market index, and official industry classification/valuation when available. Every return names its window and benchmark; missing industry valuation remains unknown.
3. Announcement classification operates only on official rows already filtered to the requested date range. Categories, direction, and risk levels are deterministic keyword rules. Unknown titles remain `other` and are never assigned a fabricated financial impact.
4. A-share analysts receive these tools automatically; non-A-share runs keep their existing tools and vendor behavior.

## Test Matrix

- China macro aliases, date cutoff, endpoint fallback, and vendor registration.
- Relative return calculations, industry valuation filtering, and source labels.
- Announcement category/risk rules, date filtering, and analyst tool bundles.
- Full regression suite plus live AkShare smoke tests on the current network.
