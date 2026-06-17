# ruff: noqa
from pathlib import Path

import pandas as pd

# 自动找到 fins-agent 项目根目录
ROOT = Path(__file__).resolve().parents[3]

# 数据路径
DATA_PATH = ROOT / "fins2026" / "week1" / "data" / "week1_assignment_data.txt"
OUT_PATH = ROOT / "fins2026" / "week1" / "data" / "week1_assignment_data.parquet"

# 1. 读取 Coke vs Pepsi 的 TSV 数据
df = pd.read_csv(DATA_PATH, sep="\t")

# 2. 把 DlyCalDt 这种 20000103 的整数日期转成真正日期
df["Date"] = pd.to_datetime(df["DlyCalDt"], format="%Y%m%d")

# 3. 保存成 Parquet
df.to_parquet(OUT_PATH)

# 4. 打印检查结果
print("Shape:")
print(df.shape)

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nUnique tickers and security names:")
print(df[["Ticker", "SecurityNm"]].drop_duplicates())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())
print("\nStep 2: Sanity checks")

print("\nDuplicate Date-Ticker rows:")
print(df.duplicated(subset=["Date", "Ticker"]).sum())

print("\nMissing values per column:")
print(df.isna().sum())

print("\nRows per ticker:")
print(df.groupby("Ticker").size())
import duckdb

print("\nStep 3: Per-ticker summary with DuckDB")

summary_sql = duckdb.sql(f"""
SELECT Ticker,
       AVG(DlyPrc) AS mean_price,
       STDDEV_POP(DlyPrc) AS sd_price,
       AVG(DlyCap) / 1000 AS mean_cap_million_usd,
       COUNT(*) AS n_days
FROM '{OUT_PATH}'
GROUP BY Ticker
ORDER BY mean_price DESC
""").df()

print(summary_sql)

print("\nStep 3: Per-ticker summary with pandas")

summary_pd = (
    df.groupby("Ticker")
    .agg(
        mean_price=("DlyPrc", "mean"),
        sd_price=("DlyPrc", lambda s: s.std(ddof=0)),
        mean_cap_million_usd=("DlyCap", lambda s: s.mean() / 1000),
        n_days=("DlyPrc", "size"),
    )
    .sort_values("mean_price", ascending=False)
    .reset_index()
)

print(summary_pd)
print("\nStep 4a: Cross-section on 2020-03-16")

crash = df[df["Date"] == "2020-03-16"].copy()
print(crash[["Date", "Ticker", "SecurityNm", "DlyPrc", "DlyRet"]])

returns = crash.set_index("Ticker")["DlyRet"]
better_ticker = returns.idxmax()
worse_ticker = returns.idxmin()
difference_pp = (returns.max() - returns.min()) * 100

print("\nBetter performer:")
print(better_ticker)

print("\nDifference in percentage points:")
print(difference_pp)


print("\nStep 4b: Time-series for KO")

ko = df[df["Ticker"] == "KO"].sort_values("Date")

print("\nFirst two rows:")
print(ko.head(2))

print("\nLast two rows:")
print(ko.tail(2))

print("\nKO shape:")
print(ko.shape)
print("\nStep 5: KO vs PEP growth of $1")

wide = df.pivot(
    index="Date",
    columns="Ticker",
    values="DlyRet"
).dropna()

growth = (1 + wide).cumprod()

final_values = growth.tail(1)
print("\nFinal dollar value:")
print(final_values)

winner = final_values.idxmax(axis=1).iloc[0]
loser = final_values.idxmin(axis=1).iloc[0]
factor = final_values[winner].iloc[0] / final_values[loser].iloc[0]

print("\nWinner:")
print(winner)

print("\nOutperformance factor:")
print(factor)

ax = growth.plot(
    title="KO vs PEP growth of $1, 2000-2025",
    logy=True,
    ylabel="value of $1 (log)"
)

fig = ax.get_figure()
fig.savefig("ko_vs_pep_growth.pdf")

print("\nSaved figure as ko_vs_pep_growth.pdf")
