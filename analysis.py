"""Credit Card Fraud Pattern Analysis (descriptive / analytical - no ML model).

Works on two inputs:
  1. data/transactions.csv   - synthetic dataset made by generate_dataset.py (default)
  2. creditcard.csv          - real dataset from Zenodo (record 7395559) / Kaggle.
                               Only Time, Amount, Class are used (V1-V28 are anonymised
                               PCA features), so the behavioural sections are skipped.

Usage:
    python analysis.py
    python analysis.py --data data/creditcard.csv --out outputs_creditcard

Outputs: <out>/figures/*.png, <out>/tables/*.csv, <out>/findings.md
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

GENUINE, FRAUD, GREY = "#4C78A8", "#E4572E", "#888888"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})


# ----------------------------------------------------------------- helpers
def load_data(path):
    df = pd.read_csv(path)
    if {"Class", "Amount", "Time"} <= set(df.columns):  # Zenodo / Kaggle format
        out = pd.DataFrame({"is_fraud": df["Class"].astype(int),
                            "amount": df["Amount"].astype(float),
                            "hour": ((df["Time"] // 3600) % 24).astype(int),
                            "day": (df["Time"] // 86400).astype(int) + 1})
        return out, "creditcard"
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.day_name()
    return df, "synthetic"


def rate_table(df, col):
    g = df.groupby(col, observed=True)["is_fraud"].agg(transactions="size", frauds="sum")
    g["fraud_rate_pct"] = g["frauds"] / g["transactions"] * 100
    return g


def chi2_p(df, col):
    return stats.chi2_contingency(pd.crosstab(df[col], df["is_fraud"]))[1]


class Out:
    def __init__(self, root):
        self.fig_dir = Path(root) / "figures"
        self.tab_dir = Path(root) / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(root)

    def fig(self, name):
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"{name}.png")
        plt.close()

    def table(self, df, name, index=True):
        df.round(3).to_csv(self.tab_dir / f"{name}.csv", index=index)


# ------------------------------------------------------------ main analysis
def main(path, out_dir):
    df, mode = load_data(path)
    o = Out(out_dir)
    F = {}  # facts for findings.md
    base_rate = df["is_fraud"].mean() * 100
    fraud, gen = df[df.is_fraud == 1], df[df.is_fraud == 0]

    # 1. class distribution ------------------------------------------------
    cd = df["is_fraud"].map({0: "Genuine", 1: "Fraud"}).value_counts().rename("transactions").to_frame()
    cd["percent"] = cd["transactions"] / len(df) * 100
    o.table(cd, "01_class_distribution")
    F.update(total=len(df), n_fraud=len(fraud), n_gen=len(gen), fraud_pct=base_rate,
             ratio=round(len(gen) / max(len(fraud), 1)))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(cd.index, cd["transactions"], color=[GENUINE, FRAUD])
    for b, (_, r) in zip(bars, cd.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(r.transactions):,}\n({r.percent:.2f}%)", ha="center", va="bottom")
    ax.set_ylim(0, cd["transactions"].max() * 1.18)
    ax.set_title("Genuine vs fraudulent transactions")
    ax.set_ylabel("Transactions")
    o.fig("01_class_distribution")

    # 2. amount analysis -----------------------------------------------------
    desc = df.groupby(df.is_fraud.map({0: "Genuine", 1: "Fraud"}))["amount"].describe(
        percentiles=[.5, .9, .99])
    o.table(desc, "02_amount_summary")
    u_p = stats.mannwhitneyu(fraud.amount, gen.amount).pvalue
    F.update(f_mean=fraud.amount.mean(), g_mean=gen.amount.mean(),
             f_med=fraud.amount.median(), g_med=gen.amount.median(), mwu_p=u_p,
             f_micro=(fraud.amount <= 10).mean() * 100, g_micro=(gen.amount <= 10).mean() * 100,
             f_high=(fraud.amount >= 200).mean() * 100, g_high=(gen.amount >= 200).mean() * 100)

    pos = df.amount[df.amount > 0]
    bins = np.logspace(np.log10(max(pos.min(), 0.5)), np.log10(pos.max()), 40)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].hist(gen.amount.clip(lower=bins[0]), bins=bins, density=True, alpha=.65, color=GENUINE, label="Genuine")
    ax[0].hist(fraud.amount.clip(lower=bins[0]), bins=bins, density=True, alpha=.65, color=FRAUD, label="Fraud")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Amount (log scale)"); ax[0].set_ylabel("Density")
    ax[0].set_title("Amount distribution"); ax[0].legend()
    bp = ax[1].boxplot([gen.amount.clip(lower=bins[0]), fraud.amount.clip(lower=bins[0])],
                       tick_labels=["Genuine", "Fraud"], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [GENUINE, FRAUD]):
        patch.set_facecolor(c); patch.set_alpha(.7)
    ax[1].set_yscale("log"); ax[1].set_title("Amount by class (outliers hidden)")
    ax[1].set_ylabel("Amount (log scale)")
    o.fig("02_amount_distribution")

    df["amount_band"] = pd.cut(df.amount, [-0.01, 10, 50, 200, 1000, np.inf],
                               labels=["<=10", "10-50", "50-200", "200-1000", "1000+"])
    ab = rate_table(df, "amount_band")
    o.table(ab, "03_fraud_rate_by_amount_band")
    F["top_band"] = ab["fraud_rate_pct"].idxmax(); F["top_band_rate"] = ab["fraud_rate_pct"].max()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(ab.index.astype(str), ab.fraud_rate_pct, color=FRAUD)
    ax.axhline(base_rate, color=GREY, ls="--", label=f"Overall {base_rate:.2f}%")
    ax.set_title("Fraud rate by transaction amount band")
    ax.set_xlabel("Amount band"); ax.set_ylabel("Fraud rate (%)"); ax.legend()
    o.fig("03_fraud_rate_by_amount_band")

    # 3. time analysis ----------------------------------------------------------
    hr = rate_table(df, "hour")
    hr["fraud_share_pct"] = hr["frauds"] / hr["frauds"].sum() * 100
    hr["genuine_share_pct"] = (hr["transactions"] - hr["frauds"]) / (hr["transactions"] - hr["frauds"]).sum() * 100
    o.table(hr, "04_hourly_pattern")
    night = df.hour.between(0, 5)
    F.update(f_night=(fraud.hour.between(0, 5)).mean() * 100, g_night=(gen.hour.between(0, 5)).mean() * 100,
             night_rate=df[night].is_fraud.mean() * 100, day_rate=df[~night].is_fraud.mean() * 100,
             peak_hour=int(hr.fraud_rate_pct.idxmax()), peak_hour_rate=hr.fraud_rate_pct.max())
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    ax[0].plot(hr.index, hr.genuine_share_pct, marker="o", color=GENUINE, label="Genuine")
    ax[0].plot(hr.index, hr.fraud_share_pct, marker="o", color=FRAUD, label="Fraud")
    ax[0].set_ylabel("% of class's transactions"); ax[0].set_title("When do transactions happen? (share by hour)")
    ax[0].legend()
    ax[1].bar(hr.index, hr.fraud_rate_pct, color=FRAUD)
    ax[1].axhline(base_rate, color=GREY, ls="--", label=f"Overall {base_rate:.2f}%")
    ax[1].set_xlabel("Hour of day"); ax[1].set_ylabel("Fraud rate (%)")
    ax[1].set_title("Fraud rate by hour"); ax[1].set_xticks(range(0, 24)); ax[1].legend()
    o.fig("04_hourly_pattern")

    if mode == "synthetic":
        daily = df.groupby("date").agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        daily["fraud_rate_pct"] = daily.frauds / daily.transactions * 100
        o.table(daily, "05_daily_pattern")
        F.update(peak_day=str(daily.frauds.idxmax()), peak_day_frauds=int(daily.frauds.max()),
                 avg_day_frauds=daily.frauds.mean(), peak_day_rate=daily.fraud_rate_pct.max())
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(pd.to_datetime(daily.index), daily.frauds, marker="o", color=FRAUD)
        ax.axhline(daily.frauds.mean(), color=GREY, ls="--", label="Daily average")
        ax.set_title("Fraud transactions per day"); ax.set_ylabel("Fraud count"); ax.legend()
        fig.autofmt_xdate()
        o.fig("05_daily_fraud_trend")

        dw = rate_table(df, "dow").reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                                           "Friday", "Saturday", "Sunday"])
        o.table(dw, "06_day_of_week")
    else:
        dd = rate_table(df, "day")
        o.table(dd, "05_day_pattern")

    # 4. behavioural patterns (synthetic dataset only) -----------------------------
    flags = {"Night (00-05h)": night, "High amount (>=200)": df.amount >= 200,
             "Small test charge (<=10)": df.amount <= 10}
    if mode == "synthetic":
        cat = rate_table(df, "merchant_category").sort_values("fraud_rate_pct", ascending=False)
        o.table(cat, "07_fraud_by_category")
        F.update(top_cat=cat.index[0], top_cat_rate=cat.fraud_rate_pct.iloc[0],
                 top3_cat=list(cat.index[:3]), cat_p=max(chi2_p(df, "merchant_category"), 1e-300))
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.barh(cat.index[::-1], cat.fraud_rate_pct[::-1], color=FRAUD)
        ax.axvline(base_rate, color=GREY, ls="--", label=f"Overall {base_rate:.2f}%")
        ax.set_xlabel("Fraud rate (%)"); ax.set_title("Fraud rate by merchant category"); ax.legend()
        o.fig("07_fraud_by_category")

        ch = rate_table(df, "channel"); fo = rate_table(df, "is_foreign")
        fo.index = fo.index.map({0: "Domestic", 1: "Foreign"})
        o.table(ch, "08_fraud_by_channel"); o.table(fo, "09_fraud_by_foreign")
        F.update(online_rate=ch.loc["online", "fraud_rate_pct"], store_rate=ch.loc["in_store", "fraud_rate_pct"],
                 foreign_rate=fo.loc["Foreign", "fraud_rate_pct"], dom_rate=fo.loc["Domestic", "fraud_rate_pct"],
                 f_online=(fraud.channel == "online").mean() * 100, g_online=(gen.channel == "online").mean() * 100)
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
        for a, t, ttl in [(ax[0], ch, "Channel"), (ax[1], fo, "Location")]:
            a.bar(t.index.astype(str), t.fraud_rate_pct, color=FRAUD)
            a.axhline(base_rate, color=GREY, ls="--"); a.set_title(f"Fraud rate by {ttl.lower()}")
            a.set_ylabel("Fraud rate (%)")
        o.fig("08_channel_and_location")

        df["velocity"] = df.txn_count_24h.clip(upper=6).astype(int).astype(str).replace("6", "6+")
        vel = rate_table(df, "velocity")
        o.table(vel, "10_fraud_by_velocity")
        F.update(f_dist=fraud.distance_from_home_km.median(), g_dist=gen.distance_from_home_km.median(),
                 f_vel=fraud.txn_count_24h.mean(), g_vel=gen.txn_count_24h.mean())
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
        bp = ax[0].boxplot([gen.distance_from_home_km, fraud.distance_from_home_km],
                           tick_labels=["Genuine", "Fraud"], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], [GENUINE, FRAUD]):
            patch.set_facecolor(c); patch.set_alpha(.7)
        ax[0].set_yscale("log"); ax[0].set_ylabel("Distance from home (km, log)")
        ax[0].set_title("Distance from home")
        ax[1].bar(vel.index, vel.fraud_rate_pct, color=FRAUD)
        ax[1].axhline(base_rate, color=GREY, ls="--")
        ax[1].set_xlabel("Transactions on the card in last 24h"); ax[1].set_ylabel("Fraud rate (%)")
        ax[1].set_title("Transaction velocity")
        o.fig("09_distance_and_velocity")

        flags.update({"Online (card-not-present)": df.channel == "online",
                      "Foreign transaction": df.is_foreign == 1,
                      "Far from home (>=50 km)": df.distance_from_home_km >= 50,
                      "High velocity (>=4 txns/24h)": df.txn_count_24h >= 4,
                      "Risky category": df.merchant_category.isin(["electronics", "jewelry", "gift_cards", "travel"])})

    # 5. rule-based suspicion indicators ----------------------------------------------
    rows = []
    for name, m in flags.items():
        m = m.astype(bool)
        caught = int(df.is_fraud[m].sum())
        rows.append({"rule": name, "flagged_txns": int(m.sum()), "flagged_pct": m.mean() * 100,
                     "frauds_caught": caught, "recall_pct": caught / len(fraud) * 100,
                     "precision_pct": caught / max(m.sum(), 1) * 100,
                     "lift": (caught / max(m.sum(), 1) * 100) / base_rate})
    rules = pd.DataFrame(rows).set_index("rule").sort_values("lift", ascending=False)
    o.table(rules, "11_single_rule_performance")
    F["best_rule"] = rules.index[0]; F["best_rule_lift"] = rules.lift.iloc[0]
    F["rules"] = rules

    fig, ax = plt.subplots(figsize=(8, 4.3))
    r = rules.iloc[::-1]
    ax.barh(r.index, r.lift, color=FRAUD)
    ax.axvline(1, color=GREY, ls="--", label="Baseline (lift = 1)")
    ax.set_xlabel("Lift = fraud rate among flagged / overall fraud rate")
    ax.set_title("How strongly does each indicator signal fraud?"); ax.legend()
    o.fig("10_rule_lift")

    df["risk_flags"] = sum(m.astype(int) for m in flags.values())
    cap = int(df.risk_flags.max())
    df["flags_bucket"] = df.risk_flags.clip(upper=min(cap, 5))
    fr = rate_table(df, "flags_bucket")
    o.table(fr, "12_fraud_rate_by_flag_count")
    thr = []
    for t in range(1, cap + 1):
        m = df.risk_flags >= t
        c = int(df.is_fraud[m].sum())
        thr.append({"flags_>=": t, "flagged_pct": m.mean() * 100, "precision_pct": c / max(m.sum(), 1) * 100,
                    "recall_pct": c / len(fraud) * 100, "lift": (c / max(m.sum(), 1) * 100) / base_rate})
    thr = pd.DataFrame(thr).set_index("flags_>=")
    o.table(thr, "13_combined_rule_thresholds")
    F["thr"] = thr
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.bar(fr.index.astype(str), fr.fraud_rate_pct, color=FRAUD)
    ax.axhline(base_rate, color=GREY, ls="--", label=f"Overall {base_rate:.2f}%")
    ax.set_xlabel("Number of risk indicators triggered"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("More red flags -> much higher fraud rate"); ax.legend()
    o.fig("11_fraud_rate_by_flag_count")

    write_findings(o.root / "findings.md", F, mode)
    print(f"Done. Mode: {mode}. Results in: {o.root}")


# ------------------------------------------------------------------ report
def write_findings(path, F, mode):
    L = []
    a = L.append
    a("# Key Findings (auto-generated by analysis.py)\n")
    a(f"Dataset mode: **{mode}**\n")
    a("## 1. Genuine vs fraud\n")
    a(f"- Total transactions: **{F['total']:,}**; genuine: **{F['n_gen']:,}**; fraud: **{F['n_fraud']:,}**")
    a(f"- Fraud percentage: **{F['fraud_pct']:.3f}%** (about 1 fraud in every {F['ratio'] + 1:,} transactions)\n")
    a("## 2. Amount behaviour\n")
    a(f"- Mean amount: fraud **{F['f_mean']:.2f}** vs genuine **{F['g_mean']:.2f}**")
    a(f"- Median amount: fraud **{F['f_med']:.2f}** vs genuine **{F['g_med']:.2f}**")
    a(f"- Amount difference is statistically significant (Mann-Whitney U p = {F['mwu_p']:.2e})")
    a(f"- Small amounts (<= 10): fraud **{F['f_micro']:.1f}%** vs genuine **{F['g_micro']:.1f}%** of transactions")
    a(f"- High amounts (>= 200): fraud **{F['f_high']:.1f}%** vs genuine **{F['g_high']:.1f}%** of transactions")
    a(f"- Riskiest amount band: **{F['top_band']}** (fraud rate {F['top_band_rate']:.2f}%)\n")
    a("## 3. Time patterns\n")
    a(f"- Night hours (00-05h) hold **{F['f_night']:.1f}%** of fraud but only **{F['g_night']:.1f}%** of genuine transactions")
    a(f"- Fraud rate at night **{F['night_rate']:.2f}%** vs daytime **{F['day_rate']:.2f}%**")
    a(f"- Peak fraud-rate hour: **{F['peak_hour']:02d}:00** ({F['peak_hour_rate']:.2f}%)")
    if mode == "synthetic":
        a(f"- Worst day: **{F['peak_day']}** with {F['peak_day_frauds']} frauds (daily average {F['avg_day_frauds']:.1f}) - a fraud burst")
    a("")
    if mode == "synthetic":
        a("## 4. Behavioural patterns\n")
        a(f"- Highest-risk categories: **{', '.join(F['top3_cat'])}** (top: {F['top_cat']} at {F['top_cat_rate']:.2f}%); chi-square p = {F['cat_p']:.2e}")
        a(f"- Online share: fraud **{F['f_online']:.1f}%** vs genuine **{F['g_online']:.1f}%**; fraud rate online {F['online_rate']:.2f}% vs in-store {F['store_rate']:.2f}%")
        a(f"- Foreign transactions: fraud rate **{F['foreign_rate']:.2f}%** vs domestic **{F['dom_rate']:.2f}%**")
        a(f"- Median distance from home: fraud **{F['f_dist']:.1f} km** vs genuine **{F['g_dist']:.1f} km**")
        a(f"- Avg transactions in last 24h: fraud **{F['f_vel']:.2f}** vs genuine **{F['g_vel']:.2f}**\n")
    a("## 5. Suspicious-transaction indicators (rule-based, no ML)\n")
    a(f"- Strongest single indicator: **{F['best_rule']}** (lift {F['best_rule_lift']:.1f}x)\n")
    a("| Rule | Flagged % | Recall % | Precision % | Lift |")
    a("|---|---|---|---|---|")
    for n, r in F["rules"].iterrows():
        a(f"| {n} | {r.flagged_pct:.1f} | {r.recall_pct:.1f} | {r.precision_pct:.2f} | {r.lift:.1f}x |")
    a("\nCombining indicators (flag a transaction when it triggers at least k rules):\n")
    a("| k | Flagged % of all txns | Precision % | Recall % | Lift |")
    a("|---|---|---|---|---|")
    for k, r in F["thr"].iterrows():
        a(f"| >= {k} | {r.flagged_pct:.2f} | {r.precision_pct:.2f} | {r.recall_pct:.1f} | {r.lift:.1f}x |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.data, a.out)
