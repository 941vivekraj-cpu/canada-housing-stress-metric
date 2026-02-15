import io
import os
import zipfile
import requests
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
OUT_DIR = "output"
OUT_FILE = os.path.join(OUT_DIR, "3610022601_household_dsr_interestonly_quarterly.csv")

PID_ZIP = "36100226"  # Table 36-10-0226-01 => zip PID drops last 2 digits

HOUSEHOLD_INCOME_LABEL = "Household income"
INTEREST_PAID_LABEL = "Interest paid"
DSR_LABEL = "Equals: debt service ratio, interest only"

# optional: keep a time window (None = keep all)
START_QUARTER = None  # e.g. "2012-01-01"
END_QUARTER = None    # e.g. "2025-12-31"


# ----------------------------
# HELPERS
# ----------------------------
def download_statcan_full_table(pid_zip: str) -> pd.DataFrame:
    url = f"https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid_zip}-eng.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()

    csvs = [n for n in names if n.lower().endswith(".csv")]
    data_candidates = [n for n in csvs if "meta" not in n.lower()]
    if not data_candidates:
        raise ValueError(f"[StatCan {pid_zip}] No data CSV found. ZIP contents: {names}")

    data_name = max(data_candidates, key=lambda n: z.getinfo(n).file_size)
    with z.open(data_name) as f:
        return pd.read_csv(f, encoding="utf-8-sig", low_memory=False)


def parse_quarter(ref_date: pd.Series) -> pd.Series:
    """
    StatCan quarterly REF_DATE often looks like: 1981Q1
    Convert to quarter-start Timestamp (e.g., 1981-01-01)
    """
    s = ref_date.astype(str).str.strip()
    # normalize formats like "1981-Q1" or "1981 Q1" to "1981Q1"
    s = s.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
    return pd.PeriodIndex(s, freq="Q").to_timestamp(how="start")


def enforce_filter(df: pd.DataFrame, estimate_label: str, uom: str, scalar_factor: str) -> pd.DataFrame:
    """
    Filter the table to:
      - specific Estimates label
      - specific UOM and SCALAR_FACTOR
    """
    out = df.loc[
        df["Estimates"].eq(estimate_label) &
        df["UOM"].eq(uom) &
        df["SCALAR_FACTOR"].astype(str).str.lower().eq(scalar_factor.lower())
    ].copy()
    return out


# ----------------------------
# MAIN
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading StatCan table 36-10-0226-01 (36100226)...")
df = download_statcan_full_table(PID_ZIP)

required_cols = {"REF_DATE", "GEO", "VALUE", "UOM", "SCALAR_FACTOR", "Estimates"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}. Columns found: {list(df.columns)}")

df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
df["Period_Quarter"] = parse_quarter(df["REF_DATE"])

# Optional time window
if START_QUARTER:
    df = df.loc[df["Period_Quarter"] >= pd.Timestamp(START_QUARTER)].copy()
if END_QUARTER:
    df = df.loc[df["Period_Quarter"] <= pd.Timestamp(END_QUARTER)].copy()

# --- Pull the 3 required series with correct units ---
# Household income: Dollars, millions
income = enforce_filter(df, HOUSEHOLD_INCOME_LABEL, uom="Dollars", scalar_factor="millions")
income = income[["Period_Quarter", "GEO", "VALUE"]].rename(
    columns={"VALUE": "HouseholdIncome_(Dollars_millions)"}
)

# Interest paid: Dollars, millions
interest = enforce_filter(df, INTEREST_PAID_LABEL, uom="Dollars", scalar_factor="millions")
interest = interest[["Period_Quarter", "GEO", "VALUE"]].rename(
    columns={"VALUE": "InterestPaid_(Dollars_millions)"}
)

# Debt service ratio (interest only): Ratio, units
dsr = enforce_filter(df, DSR_LABEL, uom="Ratio", scalar_factor="units")
dsr = dsr[["Period_Quarter", "GEO", "VALUE"]].rename(
    columns={"VALUE": "DebtServiceRatio_InterestOnly_(ratio_units)"}
)

# --- Merge into 5-column fact ---
out = (
    income.merge(interest, on=["Period_Quarter", "GEO"], how="outer")
          .merge(dsr, on=["Period_Quarter", "GEO"], how="outer")
          .sort_values(["GEO", "Period_Quarter"])
          .reset_index(drop=True)
)

# Guardrail: ensure one row per Quarter+GEO (if not, there are unexpected duplicates)
dups = out.duplicated(["Period_Quarter", "GEO"]).sum()
if dups:
    raise ValueError(f"Duplicate Quarter+GEO rows found after merge: {dups}. This means the filters need tightening.")

out.to_csv(OUT_FILE, index=False)

print("\n✅ Wrote:", OUT_FILE)
print("Rows:", len(out))
print("Quarter range:", out["Period_Quarter"].min(), "to", out["Period_Quarter"].max())
print("\nNull rates:")
print(out.isna().mean().sort_values(ascending=False).head(10))
print("\nSample:")
print(out.head(12))
