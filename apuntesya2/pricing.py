"""Centralized pricing/commission rules for ApuntesYa.

Current policy (Option B + payment methods limited):

- Seller inputs NET (what they want to receive): X
- Buyer-facing published price is computed to guarantee that net, while
  reserving a *minimum* platform margin.

The published price is computed using estimated payment processor fee
(Mercado Pago) + a minimum platform margin. At payment time, the seller is
credited *exactly* the net, and the platform keeps the remainder.

Rounding:
- Published prices are ALWAYS rounded UP (ceiling) to 1 decimal.
- UI displays prices with 1 decimal.

Environment knobs:
- MP_FEE_IMMEDIATE_TOTAL_PCT: estimated MP fee percent (e.g. 4.3)
- APY_PLATFORM_MARGIN_PCT: platform minimum margin percent (default 15)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP


def _pct_env(name: str, default_pct: str) -> Decimal:
    """Read percent env var (e.g. '15' or '4.3') and return Decimal rate."""
    raw = os.getenv(name, default_pct)
    try:
        d = Decimal(str(raw).strip())
    except Exception:
        d = Decimal(default_pct)
    return (d / Decimal(100)).quantize(Decimal("0.0001"))


# ----------------------------
# Rates (single source of truth)
# ----------------------------

# Estimated MP fee for immediate settlement (used for publishing)
MP_RATE = _pct_env("MP_FEE_IMMEDIATE_TOTAL_PCT", "8")

# Platform minimum margin
APY_RATE = _pct_env("APY_PLATFORM_MARGIN_PCT", "15")

TOTAL_RATE = (MP_RATE + APY_RATE).quantize(Decimal("0.0001"))
SELLER_SHARE = (Decimal("1") - TOTAL_RATE).quantize(Decimal("0.0001"))


def _d(x: int | float | str | Decimal) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def ceil_to_1_decimal(value: int | float | str | Decimal) -> Decimal:
    """Ceiling rounding to 1 decimal place."""
    v = _d(value)
    return (v * Decimal(10)).to_integral_value(rounding=ROUND_CEILING) / Decimal(10)


def money_1_decimal(value: int | float | str | Decimal) -> Decimal:
    """Quantize to 1 decimal for display/consistency (half up)."""
    return _d(value).quantize(Decimal("0.0"), rounding=ROUND_HALF_UP)


def amount_to_cents(amount: int | float | str | Decimal) -> int:
    """Convert ARS amount (Decimal) to integer cents."""
    a = _d(amount)
    return int((a * Decimal(100)).to_integral_value(rounding=ROUND_HALF_UP))


def cents_to_amount(cents: int | None) -> Decimal:
    """Convert integer cents to ARS Decimal amount."""
    if not cents:
        return Decimal("0")
    return _d(cents) / Decimal(100)


def published_from_net(net_amount: int | float | str | Decimal) -> Decimal:
    """Given seller net (X), return published price (P) rounded UP to 1 decimal."""
    net = _d(net_amount)
    if net <= 0:
        return Decimal("0")
    return ceil_to_1_decimal(net / SELLER_SHARE)


def published_from_net_cents(net_cents: int | None) -> int:
    """Given seller net in cents, return published price in cents."""
    net = cents_to_amount(int(net_cents or 0))
    pub = published_from_net(net)
    return amount_to_cents(pub)


@dataclass(frozen=True)
class FeeBreakdown:
    published: Decimal
    seller_net: Decimal
    platform_fee: Decimal
    mp_fee: Decimal
    total_fee: Decimal


def breakdown_from_published(published_amount: int | float | str | Decimal) -> FeeBreakdown:
    """Compute breakdown from a published price P (amount, not cents)."""
    p = _d(published_amount)
    if p <= 0:
        z = Decimal("0")
        return FeeBreakdown(z, z, z, z, z)

    seller = p * SELLER_SHARE
    plat = p * APY_RATE
    mp = p * MP_RATE
    total = p * TOTAL_RATE
    # UI: show 1 decimal
    return FeeBreakdown(
        published=money_1_decimal(p),
        seller_net=money_1_decimal(seller),
        platform_fee=money_1_decimal(plat),
        mp_fee=money_1_decimal(mp),
        total_fee=money_1_decimal(total),
    )


def breakdown_from_net(net_amount: int | float | str | Decimal) -> FeeBreakdown:
    """Compute breakdown from seller net X (amount)."""
    p = published_from_net(net_amount)
    return breakdown_from_published(p)
