import io
import os
import zipfile
import requests
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def download_statcan_full_table(pid: str) -> pd.DataFrame:
    """
    Downloads StatCan 'full table download' ZIP and returns the main data CSV as a DataFrame.
    URL pattern: https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid}-eng.zip
    """
    url = f"https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid}-eng.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()

    # Pick the largest CSV that is NOT metadata (most reliable)
    csvs = [n for n in names if n.lower().endswith(".csv")]
    data_candidates = [n for n in csvs if "meta" not in n.lower()]

    if not data_candidates:
        raise ValueError(f"[StatCan {pid}] No data CSV found. ZIP contents: {names}")

    data_name = max(data_candidates, key=lambda n: z.getinfo(n).file_size)

    with z.open(data_name) as f:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

    return df


def to_month_start(ref_date: pd.Series) -> pd.Series:
    return pd.to_datetime(ref_date, errors="coerce").dt.to_period("M").dt.to_timestamp()


def to_quarter_start_from_date(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.to_period("Q").dt.to_timestamp()


def find_col(df: pd.DataFrame, contains_text: str):
    """Find a column whose name contains a substring (case-insensitive)."""
    contains_text = contains_text.lower()
    for c in df.columns:
        if contains_text in str(c).lower():
            return c
    return None


def exact_filter(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    """Exact match filter (case-sensitive)."""
    return df[df[col].eq(value)].copy()


def numeric_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df


def assert_unique_key(df: pd.DataFrame, keys, name: str):
    d = df.duplicated(keys).sum()
    if d > 0:
        raise ValueError(f"{name}: duplicate rows on keys {keys}: {d}")


# ----------------------------
# 1) Download raw tables
# ----------------------------
income_raw = download_statcan_full_table("36100663")   # 36-10-0663-01 (quarterly income distributions)
cpi_raw    = download_statcan_full_table("18100004")   # 18-10-0004-01 (monthly CPI)
unemp_raw  = download_statcan_full_table("14100287")   # 14-10-0287-01 (monthly LFS)
mort_nb_raw= download_statcan_full_table("33100530")   # 33-10-0530-02 (quarterly non-bank mortgages by province)

# ----------------------------
# 2) Bank of Canada Valet JSON
# ----------------------------
def download_boc_series_json(series: str) -> pd.DataFrame:
    url = f"https://www.bankofcanada.ca/valet/observations/{series}/json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()

    rows = []
    for obs in j.get("observations", []):
        dt = obs.get("d")
        v = None
        if series in obs and isinstance(obs[series], dict):
            v = obs[series].get("v")
        rows.append({"date": dt, series: v})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series] = pd.to_numeric(df[series], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


prime = download_boc_series_json("V80691311")
mort5 = download_boc_series_json("V80691335")

# Convert rates to Quarter and take quarter-end (last value in quarter)
prime["Quarter"] = prime["date"].dt.to_period("Q").dt.to_timestamp()
mort5["Quarter"]  = mort5["date"].dt.to_period("Q").dt.to_timestamp()

rates_q = (prime.groupby("Quarter", as_index=False)["V80691311"].last()
           .merge(mort5.groupby("Quarter", as_index=False)["V80691335"].last(),
                  on="Quarter", how="outer")
           .rename(columns={
               "V80691311": "PrimeRate_QEnd_Pct",
               "V80691335": "Mortgage5YPosted_QEnd_Pct"
           })
           .sort_values("Quarter")
           .reset_index(drop=True))

# ----------------------------
# 3) CPI: monthly -> quarterly avg (All-items, one base)
# ----------------------------
cpi = numeric_value(cpi_raw)

# Needed columns (standard)
# REF_DATE, GEO, VALUE, UOM plus dimension col "Products and product groups"
prod_col = "Products and product groups"
if prod_col not in cpi.columns:
    raise ValueError(f"CPI: Expected column '{prod_col}' not found. Columns: {list(cpi.columns)}")

# Choose one CPI base
CPI_BASE = "2002=100"  # you can change to another base if needed

cpi["Month"] = to_month_start(cpi["REF_DATE"])
cpi["Quarter"] = cpi["Month"].dt.to_period("Q").dt.to_timestamp()

cpi_all = cpi.loc[
    cpi[prod_col].eq("All-items") &
    cpi["UOM"].eq(CPI_BASE) &
    (~cpi["GEO"].eq("Canada")),
    ["GEO", "Quarter", "VALUE"]
].rename(columns={"GEO": "Province", "VALUE": "CPI_QAvg_Source"})

# Quarterly average CPI
cpi_q = (cpi_all.groupby(["Province", "Quarter"], as_index=False)["CPI_QAvg_Source"].mean()
         .rename(columns={"CPI_QAvg_Source": "CPI_QAvg"}))

# YoY at quarterly grain
cpi_q = cpi_q.sort_values(["Province", "Quarter"])
cpi_q["CPI_YoY_Pct"] = cpi_q.groupby("Province")["CPI_QAvg"].pct_change(4) * 100

assert_unique_key(cpi_q, ["Province", "Quarter"], "CPI_Q")

# ----------------------------
# 4) Unemployment: monthly -> quarterly avg
# ----------------------------
unemp = numeric_value(unemp_raw)
unemp["Month"] = to_month_start(unemp["REF_DATE"])
unemp["Quarter"] = unemp["Month"].dt.to_period("Q").dt.to_timestamp()

# The key dimension column name can vary slightly; try to locate it
# Common names: "Labour force characteristics", "Labour force characteristics (x)"
lfs_col = None
if "Labour force characteristics" in unemp.columns:
    lfs_col = "Labour force characteristics"
else:
    lfs_col = find_col(unemp, "labour force characteristics")

if not lfs_col:
    raise ValueError(f"UNEMP: Could not find labour characteristics column. Columns: {list(unemp.columns)}")

# Filter to unemployment rate + provinces (typical useful defaults)
# Some tables also have Sex/Age group/Data type/Seasonal adjustment columns.
# We'll apply them only if they exist.
filters = [
    (lfs_col, "Unemployment rate"),
]

optional_exact_filters = {
    "Sex": "Both sexes",
    "Age group": "15 years and over",
    "Data type": "Seasonally adjusted",
    "UOM": "Percent"
}

tmp = unemp.copy()
for col, val in filters:
    if col in tmp.columns:
        tmp = tmp[tmp[col].eq(val)]

for col, val in optional_exact_filters.items():
    if col in tmp.columns:
        tmp = tmp[tmp[col].eq(val)]

tmp = tmp.loc[~tmp["GEO"].eq("Canada"), ["GEO", "Quarter", "VALUE"]]
tmp = tmp.rename(columns={"GEO": "Province", "VALUE": "UnemploymentRate_Monthly"})

unemp_q = (tmp.groupby(["Province", "Quarter"], as_index=False)["UnemploymentRate_Monthly"].mean()
           .rename(columns={"UnemploymentRate_Monthly": "Unemployment_QAvg"}))

assert_unique_key(unemp_q, ["Province", "Quarter"], "UNEMP_Q")

# ----------------------------
# 5) Income: keep quarterly disposable income by province
# ----------------------------
income = numeric_value(income_raw)

# Many StatCan income distribution tables have extra dimensions.
# We'll:
# - convert REF_DATE to Quarter
# - keep provinces (exclude Canada)
# - filter to "Disposable income" by exact match in a detected concept column

income["Quarter"] = to_quarter_start_from_date(income["REF_DATE"])

# Candidate concept columns to search (you can add/remove if your file differs)
candidate_concept_cols = [
    "Household sector transactions",
    "Household economic accounts",
    "Estimates",
    "Income, consumption and saving",
    "Income"
]

concept_col = None
for c in candidate_concept_cols:
    if c in income.columns:
        concept_col = c
        break

if concept_col is None:
    # fallback: find any column name containing "income" or "transactions"
    concept_col = find_col(income, "transactions") or find_col(income, "income")

if concept_col is None:
    raise ValueError(f"INCOME: Could not identify the concept column to filter disposable income. Columns: {list(income.columns)}")

# Exact match value to target (this may vary slightly by table)
DISPOSABLE_LABELS = [
    "Disposable income",
    "Household disposable income",
    "Disposable income, households"
]

income_prov = income.loc[~income["GEO"].eq("Canada")].copy()

# find which label exists in that column
avail = set(income_prov[concept_col].dropna().unique())
label = next((x for x in DISPOSABLE_LABELS if x in avail), None)
if label is None:
    # If labels differ, print top values to help you pick fast
    sample = list(sorted(list(avail)))[:40]
    raise ValueError(
        f"INCOME: None of expected labels found in column '{concept_col}'.\n"
        f"Expected one of: {DISPOSABLE_LABELS}\n"
        f"Sample values in '{concept_col}': {sample}"
    )

income_prov = income_prov.loc[income_prov[concept_col].eq(label), ["GEO", "Quarter", "VALUE"]]
income_q = (income_prov.rename(columns={"GEO": "Province", "VALUE": "DisposableIncome_Q"})
            .groupby(["Province", "Quarter"], as_index=False)["DisposableIncome_Q"].sum())

assert_unique_key(income_q, ["Province", "Quarter"], "INCOME_Q")

# ----------------------------
# 6) Non-bank mortgages: quarterly by province (pick "Outstanding")
# ----------------------------
mort = numeric_value(mort_nb_raw)
mort["Quarter"] = to_quarter_start_from_date(mort["REF_DATE"])

# Try to find a concept/measure column containing "Outstanding"
measure_col = find_col(mort, "outstanding") or find_col(mort, "mortgages")

# If the table has a column that explicitly says the measure (outstanding/extended/arrears),
# it will usually be an object column with those words. We’ll filter rows that contain "outstanding".
obj_cols = [c for c in mort.columns if mort[c].dtype == "object"]

def row_contains_any(row, term: str) -> bool:
    t = term.lower()
    text = " ".join(str(row[c]) for c in obj_cols).lower()
    return t in text

mort_prov = mort.loc[~mort["GEO"].eq("Canada")].copy()

# Keep rows that mention "outstanding" anywhere in object columns
out_mask = mort_prov.apply(lambda r: row_contains_any(r, "outstanding"), axis=1)

mort_q = (mort_prov.loc[out_mask, ["GEO", "Quarter", "VALUE"]]
          .rename(columns={"GEO": "Province", "VALUE": "NonBankMortgagesOutstanding_Q"})
          .groupby(["Province", "Quarter"], as_index=False)["NonBankMortgagesOutstanding_Q"].sum())

assert_unique_key(mort_q, ["Province", "Quarter"], "MORT_Q")

# ----------------------------
# 7) Merge into one Province x Quarter fact table
# ----------------------------
fact = (income_q
        .merge(cpi_q, on=["Province", "Quarter"], how="left")
        .merge(unemp_q, on=["Province", "Quarter"], how="left")
        .merge(mort_q, on=["Province", "Quarter"], how="left")
        .merge(rates_q, on="Quarter", how="left")
        .sort_values(["Province", "Quarter"])
        .reset_index(drop=True))

# Example engineered metric (proxy)
# InterestCostProxy_Q = outstanding balance * mortgage rate (%)
fact["InterestCostProxy_Q"] = (
    fact["NonBankMortgagesOutstanding_Q"] * (fact["Mortgage5YPosted_QEnd_Pct"] / 100.0)
)

fact["InterestServiceRatioProxy_Q"] = fact["InterestCostProxy_Q"] / fact["DisposableIncome_Q"]

# ----------------------------
# 8) Save outputs
# ----------------------------
os.makedirs("output", exist_ok=True)
fact.to_csv("output/fact_household_stress_quarterly.csv", index=False)

print("✅ Wrote: output/fact_household_stress_quarterly.csv")
print("Fact rows:", len(fact))
print("Duplicate Province+Quarter keys:", fact.duplicated(["Province", "Quarter"]).sum())
print("Most recent quarter:", fact["Quarter"].max())
