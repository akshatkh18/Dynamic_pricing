"""Bounded revenue search, with identical assumptions across all price scenarios."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PricingResult:
    curve: pd.DataFrame
    comparison: pd.DataFrame
    optimal_price: float
    boundary: bool
    uplift_pct: float | None


def optimize_price(service, *, stock_code, date, n_txn, elasticity, price_sensitive,
                   current_price, simulated_price, min_price, max_price, points=101):
    values = np.asarray([current_price, simulated_price, min_price, max_price], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or min_price > max_price:
        raise ValueError("Prices must be positive and the search bounds must be ordered.")
    if not isinstance(points, int) or not 2 <= points <= 1001:
        raise ValueError("Use 2 to 1001 search points.")
    candidates = np.unique(np.round(np.linspace(min_price, max_price, points), 2))
    candidates = candidates[(candidates >= min_price) & (candidates <= max_price) & (candidates > 0)]
    # Include exact baselines when in bounds, so search cannot overlook them.
    candidates = np.unique(np.r_[candidates, [v for v in values[:2] if min_price <= v <= max_price]])
    if not candidates.size:
        raise ValueError("Search bounds contain no usable price points.")
    prices = np.unique(np.r_[candidates, current_price, simulated_price])
    demand = np.asarray(service.predict(stock_code, prices, date, n_txn, elasticity, price_sensitive))
    if demand.shape != prices.shape or not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Invalid demand predictions.")
    curve = pd.DataFrame({"price": prices, "demand": demand, "revenue": prices * demand})
    if not np.isfinite(curve.revenue).all():
        raise ValueError("Revenue exceeds the supported numeric range.")
    eligible = curve[curve.price.isin(candidates)]
    # Deterministic ties: prefer the lowest price.
    best = eligible.loc[eligible.revenue.idxmax()]
    lookup = curve.set_index("price")
    comparison = pd.DataFrame([
        {"scenario": label, "price": price, **lookup.loc[price].to_dict()}
        for label, price in [("Baseline", current_price), ("Simulation", simulated_price),
                             ("Recommended", float(best.price))]
    ]).set_index("scenario")
    baseline = comparison.loc["Baseline", "revenue"]
    uplift = float((best.revenue / baseline - 1) * 100) if baseline > 0 else None
    return PricingResult(eligible.reset_index(drop=True), comparison, float(best.price),
                         bool(best.price in (candidates[0], candidates[-1])), uplift)
