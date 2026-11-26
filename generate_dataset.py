"""Generate a reproducible, synthetic credit-card transactions dataset.

The data is SYNTHETIC (no real cardholders). Fraud behaviour is modelled on
patterns commonly reported in fraud literature: night-time activity, online
(card-not-present) purchases, small "card testing" charges, high-value
purchases in resale-friendly categories, foreign usage and rapid bursts of
transactions.

Usage:
    python generate_dataset.py                  # 100,000 rows -> data/transactions.csv
    python generate_dataset.py --rows 50000 --fraud-rate 0.01 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "restaurant", "fuel", "retail", "electronics", "travel",
              "entertainment", "health", "jewelry", "gift_cards", "utilities"]
GENUINE_CAT_P = [0.24, 0.17, 0.10, 0.18, 0.05, 0.04, 0.07, 0.06, 0.01, 0.02, 0.06]
FRAUD_CAT_P = [0.03, 0.03, 0.02, 0.10, 0.30, 0.14, 0.05, 0.01, 0.12, 0.18, 0.02]

# Typical ticket-size multiplier for genuine spend per category
AMOUNT_MULT = {"grocery": 1.5, "restaurant": 1.0, "fuel": 1.3, "retail": 1.2,
               "electronics": 3.0, "travel": 4.0, "entertainment": 1.0,
               "health": 1.5, "jewelry": 4.0, "gift_cards": 1.2, "utilities": 2.0}

GENUINE_HOUR_W = [1, 1, .5, .5, .5, 1, 2, 4, 6, 7, 7, 8, 9, 8, 7, 7, 7, 8, 9, 9, 8, 6, 4, 2]
FRAUD_HOUR_W = [9, 9, 10, 10, 8, 6, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 7]

START = pd.Timestamp("2024-01-01")
DAYS = 30
BURST_DAYS = (11, 12)  # an "attack burst": fraud is 3x more likely on these days


def _p(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum()


def _timestamps(rng, n, hour_w, day_w):
    day = rng.choice(DAYS, size=n, p=_p(day_w))
    hour = rng.choice(24, size=n, p=_p(hour_w))
    sec = rng.integers(0, 3600, size=n)
    return START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") \
        + pd.to_timedelta(sec, unit="s")


def generate(rows, fraud_rate, seed):
    rng = np.random.default_rng(seed)
    n_fraud = int(round(rows * fraud_rate))
    n_gen = rows - n_fraud

    # ---------------- genuine ----------------
    cat_g = rng.choice(CATEGORIES, size=n_gen, p=_p(GENUINE_CAT_P))
    amt_g = rng.lognormal(3.3, 0.9, n_gen) * np.array([AMOUNT_MULT[c] for c in cat_g])
    dow_w = np.array([1.0] * DAYS)
    weekend = np.array([(START + pd.Timedelta(days=d)).dayofweek >= 5 for d in range(DAYS)])
    dow_w[weekend] = 1.15
    genuine = pd.DataFrame({
        "timestamp": _timestamps(rng, n_gen, GENUINE_HOUR_W, dow_w),
        "customer_id": rng.integers(1, 5001, n_gen),
        "amount": np.clip(amt_g, 1, 5000).round(2),
        "merchant_category": cat_g,
        "channel": rng.choice(["in_store", "online"], size=n_gen, p=[0.65, 0.35]),
        "is_foreign": (rng.random(n_gen) < 0.03).astype(int),
        "distance_from_home_km": rng.lognormal(1.5, 1.0, n_gen),
        "txn_count_24h": 1 + rng.poisson(1.0, n_gen),
        "is_fraud": 0,
    })

    # ---------------- fraud ----------------
    day_w_f = np.ones(DAYS)
    for d in BURST_DAYS:
        day_w_f[d] = 3.0
    test_charge = rng.random(n_fraud) < 0.35  # small "card testing" charges
    amt_f = np.where(test_charge, rng.uniform(1, 10, n_fraud),
                     np.clip(rng.lognormal(5.2, 0.8, n_fraud), 20, 5000))
    fraud = pd.DataFrame({
        "timestamp": _timestamps(rng, n_fraud, FRAUD_HOUR_W, day_w_f),
        "customer_id": rng.integers(1, 5001, n_fraud),
        "amount": amt_f.round(2),
        "merchant_category": rng.choice(CATEGORIES, size=n_fraud, p=_p(FRAUD_CAT_P)),
        "channel": rng.choice(["in_store", "online"], size=n_fraud, p=[0.20, 0.80]),
        "is_foreign": (rng.random(n_fraud) < 0.22).astype(int),
        "distance_from_home_km": rng.lognormal(3.5, 1.5, n_fraud),
        "txn_count_24h": 1 + rng.poisson(3.0, n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([genuine, fraud], ignore_index=True)
    # foreign transactions happen far from home
    far = df["is_foreign"] == 1
    df.loc[far, "distance_from_home_km"] += rng.uniform(500, 8000, far.sum())
    df["distance_from_home_km"] = df["distance_from_home_km"].round(1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--fraud-rate", type=float, default=0.012,
                    help="fraction of fraud rows (real world is ~0.1-0.2%%; raised here "
                         "so each pattern has enough fraud samples)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transactions.csv")
    a = ap.parse_args()

    df = generate(a.rows, a.fraud_rate, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows ({df.is_fraud.sum():,} fraud) -> {out}")
