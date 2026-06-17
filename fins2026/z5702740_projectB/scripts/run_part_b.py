

"""Build Project B funds, sentiment outputs, figures, and PDF report.

Run from the Project B root:
    python scripts/run_part_b.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_access  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
DATA = RESULTS / "data"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
REPORT = PROJECT_ROOT / "report"

ANNUAL_DAYS = 252
LOOKBACK = 126


def make_dirs() -> None:
    for folder in [DATA, TABLES, FIGURES, REPORT]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_panel(raw: pd.DataFrame, asset_family: str) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    date_col = "date"
    ticker_col = "ticker"
    price_col = "adjClose" if "adjClose" in df.columns else "close"

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    out["asset"] = df[ticker_col].astype(str)
    out["asset_family"] = asset_family
    out["sector"] = df["sector"].astype(str) if "sector" in df.columns else asset_family.title()
    out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    out = out.dropna(subset=["date", "asset", "price"]).sort_values(["asset", "date"])
    out["return"] = out.groupby("asset")["price"].pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["return"])
    return out[out["date"] <= pd.Timestamp("2023-12-31")]


def build_return_matrix(eq: pd.DataFrame, cr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.concat([eq, cr], ignore_index=True)
    meta = panel[["asset", "asset_family", "sector"]].drop_duplicates("asset")
    matrix = panel.pivot_table(index="date", columns="asset", values="return", aggfunc="mean").sort_index()
    equity_calendar = pd.Index(sorted(eq["date"].dropna().unique()))
    matrix = matrix.reindex(equity_calendar)
    matrix = matrix.loc[:, matrix.isna().mean() <= 0.25].fillna(0.0)
    meta = meta[meta["asset"].isin(matrix.columns)].copy()
    return matrix, meta


def equal_weights(n: int) -> np.ndarray:
    return np.repeat(1 / n, n)


def opt_weights(train: pd.DataFrame, method: str) -> np.ndarray:
    n = train.shape[1]
    if n <= 1 or method == "Equal Weight":
        return equal_weights(n)

    mu = train.mean().to_numpy()
    cov = train.cov().to_numpy()
    cov = np.nan_to_num(cov) + np.eye(n) * 1e-6
    x0 = equal_weights(n)
    bounds = [(0.0, 0.20)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    if method == "Minimum Variance":
        objective = lambda w: float(w @ cov @ w)
    elif method == "Maximum Sharpe":
        def objective(w: np.ndarray) -> float:
            vol = math.sqrt(max(float(w @ cov @ w), 1e-12))
            return -float(w @ mu) / vol
    else:
        return x0

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        return x0
    weights = np.maximum(res.x, 0)
    return weights / weights.sum()


def backtest(ret: pd.DataFrame, family: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = ret.index
    rebalance = pd.Series(dates, index=dates).groupby(pd.Index(dates).to_period("M")).first().tolist()
    rebalance = [pd.Timestamp(d) for d in rebalance if dates.get_loc(d) >= LOOKBACK]
    current: dict[str, pd.Series] = {}
    fund_rows: list[dict] = []
    weight_rows: list[dict] = []

    for date in dates[LOOKBACK:]:
        i = dates.get_loc(date)
        if not current or pd.Timestamp(date) in rebalance:
            train = ret.iloc[i - LOOKBACK:i]
            for method in methods:
                w = pd.Series(opt_weights(train, method), index=train.columns)
                current[method] = w
                for asset, weight in w.items():
                    weight_rows.append({"date": date, "fund": f"{family} {method}", "asset": asset, "weight": float(weight)})

        today = ret.loc[date]
        for method, weights in current.items():
            r = float((today.reindex(weights.index).fillna(0.0) * weights).sum())
            fund_rows.append({"date": date, "fund": f"{family} {method}", "return": r})

    return pd.DataFrame(fund_rows), pd.DataFrame(weight_rows)


def add_growth_and_drawdown(fund_returns: pd.DataFrame) -> pd.DataFrame:
    output = []
    for fund, g in fund_returns.groupby("fund"):
        g = g.sort_values("date").copy()
        g["growth_of_1"] = (1 + g["return"]).cumprod()
        g["drawdown"] = g["growth_of_1"] / g["growth_of_1"].cummax() - 1
        output.append(g)
    return pd.concat(output, ignore_index=True)


def metrics(fund_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fund, g in fund_returns.groupby("fund"):
        r = g.sort_values("date")["return"].astype(float)
        growth = (1 + r).cumprod()
        ann_ret = float(growth.iloc[-1] ** (ANNUAL_DAYS / len(r)) - 1)
        ann_vol = float(r.std(ddof=1) * math.sqrt(ANNUAL_DAYS))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        dd = growth / growth.cummax() - 1
        rows.append({
            "fund": fund,
            "annualised_return": ann_ret,
            "annualised_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(dd.min()),
            "total_return": float(growth.iloc[-1] - 1),
            "final_growth_of_1": float(growth.iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)


def sentiment_score(text: str) -> float:
    pos = {"beat", "growth", "gain", "rise", "rally", "surge", "profit", "upgrade", "strong", "record", "bullish", "recovery", "higher"}
    neg = {"miss", "loss", "fall", "drop", "plunge", "downgrade", "weak", "lawsuit", "fraud", "bearish", "risk", "cut", "slump", "decline", "lower"}
    words = [w.strip(".,:;!?()[]{}'\"").lower() for w in str(text).split()]
    p = sum(w in pos for w in words)
    n = sum(w in neg for w in words)
    return (p - n) / max(p + n, 1)


def build_sentiment(eq: pd.DataFrame) -> pd.DataFrame:
    try:
        news = data_access.load_news_headlines()
    except Exception:
        news = pd.DataFrame()

    if news.empty or "title" not in news.columns:
        dates = sorted(eq["date"].unique())
        sectors = sorted(eq["sector"].dropna().unique())
        rows = [{"date": d, "sector": s, "sentiment": 0.0, "lagged_sentiment": 0.0, "headline_count": 0} for d in dates for s in sectors]
        return pd.DataFrame(rows)

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce", utc=True).dt.tz_localize(None)
    news = news.dropna(subset=["date"])
    news["sector"] = news["sector"].astype(str) if "sector" in news.columns else "Unknown"
    news["sentiment"] = news["title"].map(sentiment_score)
    out = news.groupby(["date", "sector"], as_index=False).agg(sentiment=("sentiment", "mean"), headline_count=("sentiment", "size"))
    out = out.sort_values(["sector", "date"])
    out["lagged_sentiment"] = out.groupby("sector")["sentiment"].shift(1).fillna(0.0)
    return out[out["date"] <= pd.Timestamp("2023-12-31")]


def sentiment_tilt(eq_returns: pd.DataFrame, meta: pd.DataFrame, sentiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_map = meta.set_index("asset")["sector"].to_dict()
    sent = sentiment.pivot_table(index="date", columns="sector", values="lagged_sentiment", aggfunc="mean").reindex(eq_returns.index).ffill().fillna(0.0)
    fund_rows = []
    weight_rows = []
    for date in eq_returns.index[LOOKBACK:]:
        scores = np.array([max(sent.loc[date].get(sector_map.get(a, ""), 0.0), 0.0) for a in eq_returns.columns])
        base = equal_weights(len(eq_returns.columns))
        tilt = scores / scores.sum() if scores.sum() > 0 else base
        weights = 0.75 * base + 0.25 * tilt
        weights = weights / weights.sum()
        ret = float((eq_returns.loc[date].to_numpy() * weights).sum())
        fund_rows.append({"date": date, "fund": "Equity Sentiment Tilt", "return": ret})
        if pd.Timestamp(date).day <= 7:
            for asset, weight in zip(eq_returns.columns, weights):
                weight_rows.append({"date": date, "fund": "Equity Sentiment Tilt", "asset": asset, "weight": float(weight)})
    return pd.DataFrame(fund_rows), pd.DataFrame(weight_rows)


def save_figures(fund_returns: pd.DataFrame, fund_weights: pd.DataFrame, sentiment: pd.DataFrame, perf: pd.DataFrame) -> None:
    fund_returns.pivot_table(index="date", columns="fund", values="growth_of_1").plot(figsize=(10, 6))
    plt.title("Growth of $1 across funds")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(FIGURES / "growth_of_1_funds.png", dpi=160)
    plt.close()

    first_fund = fund_returns["fund"].iloc[0]
    fund_returns[fund_returns["fund"] == first_fund].plot(x="date", y="drawdown", figsize=(10, 5), legend=False)
    plt.title(f"Drawdown for {first_fund}")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(FIGURES / "drawdown_example.png", dpi=160)
    plt.close()

    latest = fund_weights.sort_values("date").groupby(["fund", "asset"], as_index=False).tail(1).sort_values("weight", ascending=False).head(20)
    latest.plot.bar(x="asset", y="weight", figsize=(11, 5), legend=False)
    plt.title("Latest portfolio weights")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(FIGURES / "latest_portfolio_weights.png", dpi=160)
    plt.close()

    perf.sort_values("sharpe_ratio").plot.barh(x="fund", y="sharpe_ratio", figsize=(10, 5), legend=False)
    plt.title("Sharpe ratios across funds")
    plt.tight_layout()
    plt.savefig(FIGURES / "sharpe_ratio_barplot.png", dpi=160)
    plt.close()

    sentiment.pivot_table(index="date", columns="sector", values="lagged_sentiment").plot(figsize=(11, 6))
    plt.title("Lagged sector sentiment index")
    plt.tight_layout()
    plt.savefig(FIGURES / "sector_sentiment_index.png", dpi=160)
    plt.close()

    compare = fund_returns[fund_returns["fund"].isin(["Equity Equal Weight", "Equity Sentiment Tilt"])]
    compare.pivot_table(index="date", columns="fund", values="growth_of_1").plot(figsize=(10, 5))
    plt.title("Fusion comparison: base vs sentiment tilt")
    plt.tight_layout()
    plt.savefig(FIGURES / "fusion_before_after.png", dpi=160)
    plt.close()


def write_report(perf: pd.DataFrame) -> None:
    with PdfPages(REPORT / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "FINS5545 Project B Report", fontsize=18, weight="bold")
        fig.text(0.08, 0.91, "Student: z5702740", fontsize=11)
        summary = (
            "This report documents a reproducible Project B baseline. The script builds equity-only, "
            "crypto-only, and combined equity-plus-crypto systematic funds. Weights are estimated using "
            "only past data in a 126-trading-day lookback window, then applied out of sample with monthly "
            "rebalancing. The sentiment index uses headline text aggregated by equity sector and lags the "
            "signal by one day to reduce look-ahead bias. The Streamlit app should load the precomputed "
            "CSV outputs rather than rerunning the full backtest."
        )
        fig.text(0.08, 0.84, summary, fontsize=10, wrap=True, va="top")
        fig.text(0.08, 0.64, "Top performance metrics", fontsize=13, weight="bold")
        fig.text(0.08, 0.61, perf.head(8).round(4).to_string(index=False), fontsize=7.5, family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    make_dirs()
    eq = clean_panel(data_access.load_equity_prices(), "equity")
    cr = clean_panel(data_access.load_crypto_prices(), "crypto")
    print("equities:", eq.shape, "crypto:", cr.shape)

    matrix, meta = build_return_matrix(eq, cr)
    equity_assets = meta.loc[meta["asset_family"] == "equity", "asset"].tolist()
    crypto_assets = meta.loc[meta["asset_family"] == "crypto", "asset"].tolist()

    combined_r, combined_w = backtest(matrix, "Combined", ["Equal Weight", "Minimum Variance", "Maximum Sharpe"])
    equity_r, equity_w = backtest(matrix[equity_assets], "Equity", ["Equal Weight"])
    crypto_r, crypto_w = backtest(matrix[crypto_assets], "Crypto", ["Equal Weight"])

    sentiment = build_sentiment(eq)
    sentiment_r, sentiment_w = sentiment_tilt(matrix[equity_assets], meta, sentiment)

    fund_returns = pd.concat([combined_r, equity_r, crypto_r, sentiment_r], ignore_index=True)
    fund_weights = pd.concat([combined_w, equity_w, crypto_w, sentiment_w], ignore_index=True)
    fund_returns = add_growth_and_drawdown(fund_returns)
    perf = metrics(fund_returns)

    fund_returns.to_csv(DATA / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA / "fund_weights.csv", index=False)
    sentiment.to_csv(DATA / "sector_sentiment_index.csv", index=False)
    perf.to_csv(TABLES / "performance_metrics.csv", index=False)
    perf[perf["fund"].isin(["Equity Equal Weight", "Equity Sentiment Tilt"])].to_csv(TABLES / "fusion_comparison.csv", index=False)

    save_figures(fund_returns, fund_weights, sentiment, perf)
    write_report(perf)

    print("Saved required Project B artifacts:")
    for path in [
        DATA / "fund_returns.csv",
        DATA / "fund_weights.csv",
        DATA / "sector_sentiment_index.csv",
        TABLES / "performance_metrics.csv",
        REPORT / "report.pdf",
    ]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
