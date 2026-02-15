import io
import os
import zipfile
import requests
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
OUT_DIR = "output"
OUT_MONTHLY = os.path.join(OUT_DIR, "mortgage_outstanding_canada_monthly_2012_2025.csv")
OUT_QUARTERLY = os.path.join(OUT_DIR, "mortgage_outstanding_canada_quarterly_2012_2025.csv")

# StatCan Table 10-10-0129-01 => PID 10100129 (ZIP uses pid without the last 2 digits)
PID = "10100129"

START_DATE = pd.Timestamp("2012-01-01")
END_DATE = pd.Timestamp("2025-12-31")

SCALAR_MAP = {
    "units": 1,
    "unit": 1,
    "thousands": 1_000,
    "thousand": 1_000,
    "millions": 1_000_000,
    "million": 1_000_000,
    "billions": 1_000_000_000,
    "billion": 1_000_000_000,
}

# ----------------------------
# HELPERS
# ----------------------------
def download_statcan_full_table(pid: str) -> pd.DataFrame:
    """
    Downloads StatCan full table ZIP:
    https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid}-eng.zip
    """
    url = f"https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid}-eng.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()

    csvs = [n for n in names if n.lower().endswith(".csv")]
    data_candidates = [n for n in csvs if "meta" not in n.lower()]
    if not data_candidates:
        raise ValueError(f"[StatCan {pid}] No data CSV found. ZIP contents: {names}")

    # pick biggest non-metadata CSV
    data_name = max(data_candidates, key=lambda n: z.getinfo(n).file_size)

    with z.open(data_name) as f:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

    return df


def to_month_start(ref_date: pd.Series) -> pd.Series:
    return pd.to_datetime(ref_date, errors="coerce").dt.to_period("M").dt.to_timestamp()


def apply_scalar_to_cad(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Converts VALUE into dollars (CAD units) using SCALAR_FACTOR if present.
    Also keeps a "million CAD" version for Tableau-friendly readability.
    """
    out = df.copy()

    if "SCALAR_FACTOR" in out.columns:
        mult = out["SCALAR_FACTOR"].astype(str).str.lower().map(SCALAR_MAP).fillna(1)
    else:
        mult = 1

    out[f"{value_col}_CAD_(Dollars_units)"] = out[value_col] * mult
    out[f"{value_col}_MillionCAD_(Dollars_millions)"] = out[f"{value_col}_CAD_(Dollars_units)"] / 1_000_000

    return out


def find_col_contains(df: pd.DataFrame, text: str) -> str | None:
    t = text.lower()
    for c in df.columns:
        if t in str(c).lower():
            return c
    return None


# ----------------------------
# MAIN
# ----------------------------
print("Downloading StatCan table 10-10-0129-01 (PID 10100129)...")
raw = download_statcan_full_table(PID)

# Normalize basics
df = raw.copy()
df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
df["Month"] = to_month_start(df["REF_DATE"])

# Date window
df = df.loc[df["Month"].notna()].copy()
df = df.loc[(df["Month"] >= START_DATE) & (df["Month"] <= END_DATE)].copy()

# Identify the component / lender breakdown column
# From StatCan catalogue: "Components" is one dimension of this table. :contentReference[oaicite:1]{index=1}
components_col = None
if "Components" in df.columns:
    components_col = "Components"
else:
    components_col = find_col_contains(df, "component")

if components_col is None:
    raise ValueError(f"Could not find a 'Components' column. Columns: {list(df.columns)}")

# Identify month timing column (e.g., "At month-end")
timing_col = find_col_contains(df, "month-end") or find_col_contains(df, "month end") or find_col_contains(df, "monthly")

# Identify seasonal adjustment column
sa_col = "Seasonal adjustment" if "Seasonal adjustment" in df.columns else find_col_contains(df, "seasonal")

# Filters (robust approach):
# - Geography should be Canada (this table is Canada-level)
# - Choose "Total outstanding balances" from Components
# - Prefer "At month-end" (if that column exists)
# - Prefer "Seasonally adjusted" (if exists) else keep unadjusted
df["GEO"] = df["GEO"].astype(str)

df = df.loc[df["GEO"].eq("Canada")].copy()

# Keep "Total outstanding balances" rows
df = df.loc[df[components_col].astype(str).str.lower().str.contains("total outstanding balances", na=False)].copy()

# If the timing column exists, keep "At month-end"
if timing_col and timing_col in df.columns:
    df = df.loc[df[timing_col].astype(str).str.lower().str.contains("month-end", na=False)].copy()

# If seasonal adjustment column exists, keep seasonally adjusted (better for trend charts)
if sa_col and sa_col in df.columns:
    # if SA exists, keep SA; if not present for some reason, fall back to all rows
    sa_mask = df[sa_col].astype(str).str.lower().str.contains("seasonally adjusted", na=False)
    if sa_mask.any():
        df = df.loc[sa_mask].copy()

# Keep only dollars rows (sometimes tables have other UOM)
if "UOM" in df.columns:
    df = df.loc[df["UOM"].astype(str).str.lower().str.contains("dollar", na=False)].copy()

# Select columns
keep_cols = ["Month", "VALUE"]
for c in ["UOM", "SCALAR_FACTOR", "SCALAR_ID", components_col]:
    if c in df.columns:
        keep_cols.append(c)

out = df[keep_cols].copy()

# Convert to CAD units + Million CAD + add units into header names
out = out.rename(columns={"VALUE": "MortgageOutstanding_Monthly_Value"})
out = apply_scalar_to_cad(out, "MortgageOutstanding_Monthly_Value")

# Clean final monthly output (Tableau friendly)
monthly = out[[
    "Month",
    "MortgageOutstanding_Monthly_Value_CAD_(Dollars_units)",
    "MortgageOutstanding_Monthly_Value_MillionCAD_(Dollars_millions)"
]].sort_values("Month").reset_index(drop=True)

# Quarterly rollup: average of months in quarter (you can change to "last" if you want quarter-end)
monthly["Quarter"] = monthly["Month"].dt.to_period("Q").dt.to_timestamp()
quarterly = (monthly.groupby("Quarter", as_index=False)
             .agg({
                 "MortgageOutstanding_Monthly_Value_CAD_(Dollars_units)": "mean",
                 "MortgageOutstanding_Monthly_Value_MillionCAD_(Dollars_millions)": "mean"
             })
             .rename(columns={
                 "MortgageOutstanding_Monthly_Value_CAD_(Dollars_units)": "MortgageOutstanding_QAvg_CAD_(Dollars_units)",
                 "MortgageOutstanding_Monthly_Value_MillionCAD_(Dollars_millions)": "MortgageOutstanding_QAvg_MillionCAD_(Dollars_millions)"
             })
             .sort_values("Quarter")
             .reset_index(drop=True))

# Add YoY for quarterly (4 quarters back)
quarterly["MortgageOutstanding_YoY_Pct"] = (
    quarterly["MortgageOutstanding_QAvg_CAD_(Dollars_units)"].pct_change(4) * 100
)

# Write files
os.makedirs(OUT_DIR, exist_ok=True)
monthly.to_csv(OUT_MONTHLY, index=False)
quarterly.to_csv(OUT_QUARTERLY, index=False)

print("\n✅ Wrote:", OUT_MONTHLY)
print("✅ Wrote:", OUT_QUARTERLY)
print("Monthly range:", monthly["Month"].min(), "to", monthly["Month"].max())
print("Quarterly range:", quarterly["Quarter"].min(), "to", quarterly["Quarter"].max())
print("Rows monthly:", len(monthly), "| rows quarterly:", len(quarterly))
