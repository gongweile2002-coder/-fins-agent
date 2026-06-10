from pathlib import Path
import pandas as pd

# =========================
# Output folders
# =========================
OUT_DIR = Path("fins2026/week2/scratch/us_macro_practice")
FIGURE_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Use local / course data instead of FRED web download
# This avoids macOS SSL certificate and encoding errors.
# =========================
POSSIBLE_INPUTS = [
    Path("fins2026/week2/data/fred_market_macro.csv"),
    Path("fins2026/week2/data/us_macro.csv"),
    Path("fins2026/week2/data/macro_panel.csv"),
    Path("fins2026/week2/scratch/us_macro_practice/fred_market_macro_typed.csv"),
    Path("fins2026/week2/scratch/fred_market_macro_typed.csv"),
]

input_path = None
for path in POSSIBLE_INPUTS:
    if path.exists():
        input_path = path
        break

if input_path is None:
    print("Could not find a local macro CSV file.")
    print("Checked these paths:")
    for path in POSSIBLE_INPUTS:
        print("-", path)
    print("\nNext action: put the course macro CSV into fins2026/week2/data/ and name it fred_market_macro.csv")
    raise FileNotFoundError("No local macro CSV file found.")

print("Reading local macro data from:")
print(input_path)

fred = pd.read_csv(input_path, encoding="latin1")

print("\nRaw data loaded")
print(fred.head())
print(fred.info())

# =========================
# Standardise column names
# =========================
if "observation_date" in fred.columns:
    fred = fred.rename(columns={"observation_date": "date"})
elif "Date" in fred.columns:
    fred = fred.rename(columns={"Date": "date"})

if "date" not in fred.columns:
    raise ValueError(f"No date column found. Columns are: {list(fred.columns)}")

fred["date"] = pd.to_datetime(fred["date"], errors="coerce")

for col in fred.columns:
    if col != "date":
        fred[col] = pd.to_numeric(fred[col].replace(".", pd.NA), errors="coerce")

print("\nAfter type conversion")
print(fred.dtypes)

print("\nRows received:")
print(len(fred))

print("\nDate range:")
print(fred["date"].min(), "to", fred["date"].max())

print("\nMissing values:")
print(fred.isna().sum())

out_csv = OUT_DIR / "fred_market_macro_typed.csv"
fred.to_csv(out_csv, index=False)

print("\nSaved typed data")
print(out_csv)

# =========================
# Stage 2: Build the five-figure US macro narrative
# =========================
import matplotlib.pyplot as plt

print("\nStage 2: building five US macro figures")

macro = fred.copy().sort_values("date")

# Keep the lecture window required by the class exercise.
macro = macro[(macro["date"] >= "2015-01-01") & (macro["date"] <= "2025-12-31")].copy()

# Use date as index for time-series plots.
macro = macro.set_index("date")

# Make sure key derived columns exist even if the fixture gives only raw series.
if "ten_year_minus_two_year" not in macro.columns and {"DGS10", "DGS2"}.issubset(macro.columns):
    macro["ten_year_minus_two_year"] = macro["DGS10"] - macro["DGS2"]

if "ten_year_minus_three_month" not in macro.columns and {"DGS10", "DTB3"}.issubset(macro.columns):
    macro["ten_year_minus_three_month"] = macro["DGS10"] - macro["DTB3"]

if "SP500_RETURN_PCT" not in macro.columns and "SP500" in macro.columns:
    macro["SP500_RETURN_PCT"] = macro["SP500"].pct_change() * 100

if "SP500_CUMULATIVE_RETURN_PCT" not in macro.columns and "SP500" in macro.columns:
    sp500_growth = macro["SP500"] / macro["SP500"].dropna().iloc[0]
    macro["SP500_CUMULATIVE_RETURN_PCT"] = (sp500_growth - 1) * 100

if "vix_rolling_21d" not in macro.columns and "VIXCLS" in macro.columns:
    macro["vix_rolling_21d"] = macro["VIXCLS"].rolling(21).mean()

macro_out = OUT_DIR / "fred_market_macro_stage2.csv"
macro.reset_index().to_csv(macro_out, index=False)
print("Saved Stage 2 data")
print(macro_out)

# Figure 1: Treasury yield panel
yield_cols = [col for col in ["DGS10", "DGS2", "DTB3"] if col in macro.columns]
ax = macro[yield_cols].plot(figsize=(9, 5))
ax.set_title("Treasury yields shifted sharply across the sample")
ax.set_xlabel("Date")
ax.set_ylabel("Yield (%)")
ax.legend(title="Series")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig1 = FIGURE_DIR / "01_treasury_yields.png"
plt.savefig(fig1, dpi=150)
plt.show()
plt.close()

# Figure 2: Yield curve inversion
curve_cols = [col for col in ["ten_year_minus_two_year", "ten_year_minus_three_month", "T10Y2Y"] if col in macro.columns]
curve_cols = list(dict.fromkeys(curve_cols))
ax = macro[curve_cols].plot(figsize=(9, 5))
ax.axhline(0, linestyle="--")
ax.set_title("Yield curve inversion signalled tighter macro conditions")
ax.set_xlabel("Date")
ax.set_ylabel("Spread, percentage points")
ax.legend(title="Spread")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2 = FIGURE_DIR / "02_yield_curve_inversion.png"
plt.savefig(fig2, dpi=150)
plt.show()
plt.close()

# Figure 3: S&P 500 growth of $1
sp500_growth = macro["SP500"] / macro["SP500"].dropna().iloc[0]
ax = sp500_growth.plot(figsize=(9, 5))
ax.set_title("S&P 500 growth of $1, 2015-2025")
ax.set_xlabel("Date")
ax.set_ylabel("Value of $1 invested")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3 = FIGURE_DIR / "03_sp500_growth_of_1.png"
plt.savefig(fig3, dpi=150)
plt.show()
plt.close()

# Figure 4: Federal funds rate vs unemployment
ax = macro[["FEDFUNDS", "UNRATE"]].plot(figsize=(9, 5))
ax.set_title("Policy rate and unemployment moved through different cycles")
ax.set_xlabel("Date")
ax.set_ylabel("Percent")
ax.legend(title="Series")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig4 = FIGURE_DIR / "04_fedfunds_unemployment.png"
plt.savefig(fig4, dpi=150)
plt.show()
plt.close()

# Figure 5: S&P 500 return vs change in VIX
scatter = macro[["SP500_RETURN_PCT", "VIXCLS"]].copy()
scatter["VIX_CHANGE"] = scatter["VIXCLS"].diff()
scatter = scatter.dropna()
ax = scatter.plot.scatter(x="VIX_CHANGE", y="SP500_RETURN_PCT", figsize=(8, 5))
ax.axhline(0, linestyle="--")
ax.axvline(0, linestyle="--")
ax.set_title("Equity returns weakened when volatility rose")
ax.set_xlabel("Daily change in VIX")
ax.set_ylabel("S&P 500 daily return (%)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig5 = FIGURE_DIR / "05_sp500_vix_scatter.png"
plt.savefig(fig5, dpi=150)
plt.show()
plt.close()

print("\nSaved five figures")
for fig_path in [fig1, fig2, fig3, fig4, fig5]:
    print(fig_path)
