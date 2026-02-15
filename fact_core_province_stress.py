import io
import os
import zipfile
import requests
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
OUT_DIR = "output"
OUT_FILE = os.path.join(OUT_DIR, "fact_core_province_quarter.csv")

# StatCan PIDs
PID_INCOME = "36100663"   # Income (quarterly)
PID_CPI    = "18100004"   # CPI (monthly)
PID_UNEMP  = "14100287"   # Unemployment (monthly)

# BoC series
SERIES_PRIME = "V80691311"
SERIES_MORT5 = "V80691335"

# CPI
CPI_BASE = "2002=100"
CPI_ALL_ITEMS_LABEL = "All-items"
CPI_SHELTER_LABELS = ["Shelter"]

# Income
DISPOSABLE_LABELS = [
    "Disposable income",
    "Household disposable income",
    "Disposable income, households",
]

UNEMP_OPTIONAL_FILTERS = {
    "Sex": "Both sexes",
    "Age group": "15 years and over",
    "Data type": "Seasonally adjusted",
    "UOM": "Percent",
}

SCALAR_MAP = {
    "units": 1,
    "thousands": 1_000,
    "millions": 1_000_000,
    "billions": 1_000_000_000,
}

# ----------------------------
# HELPERS
# ----------------------------
def download_statcan_full_table(pid: str) -> pd.DataFrame:
    url = f"https://www150.statcan.gc.ca/n1/en/tbl/csv/{pid}-eng.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()

    csvs = [n for n in names if n.lower().endswith(".csv")]
    data_candidates = [n for n in csvs if "meta" not in n.lower()]
    if not data_candidates:
        raise ValueError(f"[StatCan {pid}] No data CSV found. ZIP contents: {names}")

    data_name = max(data_candidates, key=lambda n: z.getinfo(n).file_size)

    with z.open(data_name) as f:
        return pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

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
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def to_month_start(ref_date: pd.Series) -> pd.Series:
    return pd.to_datetime(ref_date, errors="coerce").dt.to_period("M").dt.to_timestamp()

def to_quarter_start_from_date(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.to_period("Q").dt.to_timestamp()

def find_col(df: pd.DataFrame, contains_text: str):
    t = contains_text.lower()
    for c in df.columns:
        if t in str(c).lower():
            return c
    return None

def numeric_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df

def apply_scalar(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    mult = out["SCALAR_FACTOR"].astype(str).str.lower().map(SCALAR_MAP).fillna(1)
    out[f"{value_col}_CAD"] = out[value_col] * mult
    return out

# ----------------------------
# DOWNLOAD
# ----------------------------
print("Downloading StatCan tables...")
income_raw = download_statcan_full_table(PID_INCOME)
cpi_raw    = download_statcan_full_table(PID_CPI)
unemp_raw  = download_statcan_full_table(PID_UNEMP)

print("Downloading Bank of Canada series...")
prime = download_boc_series_json(SERIES_PRIME)
mort5 = download_boc_series_json(SERIES_MORT5)

# ----------------------------
# BoC -> Quarter-end
# ----------------------------
prime["Quarter"] = prime["date"].dt.to_period("Q").dt.to_timestamp()
mort5["Quarter"] = mort5["date"].dt.to_period("Q").dt.to_timestamp()

rates_q = (
    prime.groupby("Quarter", as_index=False)[SERIES_PRIME].last()
    .merge(mort5.groupby("Quarter", as_index=False)[SERIES_MORT5].last(), on="Quarter", how="outer")
    .rename(columns={
        SERIES_PRIME: "PrimeRate_QEnd_Pct",
        SERIES_MORT5: "Mortgage5YPosted_QEnd_Pct",
    })
    .sort_values("Quarter")
)

# ----------------------------
# CPI monthly -> quarterly avg
# Filters:
# - UOM == 2002=100
# - Products and product groups == All-items / Shelter
# - GEO != Canada
# ----------------------------
cpi = numeric_value(cpi_raw)
prod_col = "Products and product groups"
if prod_col not in cpi.columns:
    raise ValueError(f"CPI: Missing '{prod_col}'. Columns: {list(cpi.columns)}")

cpi["Month"] = to_month_start(cpi["REF_DATE"])
cpi["Quarter"] = cpi["Month"].dt.to_period("Q").dt.to_timestamp()
cpi["Province"] = cpi["GEO"].astype(str).str.strip()

cpi = cpi.loc[(cpi["UOM"] == CPI_BASE) & (cpi["GEO"] != "Canada")].copy()

cpi_all = cpi.loc[cpi[prod_col] == CPI_ALL_ITEMS_LABEL, ["Province", "Quarter", "VALUE"]]
cpi_all_q = (
    cpi_all.groupby(["Province", "Quarter"], as_index=False)["VALUE"].mean()
    .rename(columns={"VALUE": "CPI_AllItems_QAvg_Index_2002eq100"})
)
cpi_all_q = cpi_all_q.sort_values(["Province", "Quarter"])
cpi_all_q["CPI_AllItems_YoY_Pct"] = cpi_all_q.groupby("Province")["CPI_AllItems_QAvg_Index_2002eq100"].pct_change(4) * 100

cpi_shel = cpi.loc[cpi[prod_col].isin(CPI_SHELTER_LABELS), ["Province", "Quarter", "VALUE"]]
if len(cpi_shel) > 0:
    cpi_shel_q = (
        cpi_shel.groupby(["Province", "Quarter"], as_index=False)["VALUE"].mean()
        .rename(columns={"VALUE": "CPI_Shelter_QAvg_Index_2002eq100"})
    )
    cpi_shel_q = cpi_shel_q.sort_values(["Province", "Quarter"])
    cpi_shel_q["CPI_Shelter_YoY_Pct"] = cpi_shel_q.groupby("Province")["CPI_Shelter_QAvg_Index_2002eq100"].pct_change(4) * 100
else:
    cpi_shel_q = pd.DataFrame(columns=["Province","Quarter","CPI_Shelter_QAvg_Index_2002eq100","CPI_Shelter_YoY_Pct"])

# ----------------------------
# Unemployment monthly -> quarterly avg
# Filters:
# - Labour force characteristics == Unemployment rate
# - optional: Sex, Age group, Data type, UOM (if present)
# - GEO != Canada
# ----------------------------
unemp = numeric_value(unemp_raw)
unemp["Month"] = to_month_start(unemp["REF_DATE"])
unemp["Quarter"] = unemp["Month"].dt.to_period("Q").dt.to_timestamp()
unemp["Province"] = unemp["GEO"].astype(str).str.strip()

lfs_col = "Labour force characteristics" if "Labour force characteristics" in unemp.columns else find_col(unemp, "labour force characteristics")
if not lfs_col:
    raise ValueError(f"UNEMP: labour force characteristics column not found. Columns: {list(unemp.columns)}")

tmp = unemp.loc[unemp[lfs_col] == "Unemployment rate"].copy()
for col, val in UNEMP_OPTIONAL_FILTERS.items():
    if col in tmp.columns:
        tmp = tmp.loc[tmp[col] == val].copy()

tmp = tmp.loc[tmp["GEO"] != "Canada", ["Province", "Quarter", "VALUE"]]
unemp_q = (
    tmp.groupby(["Province", "Quarter"], as_index=False)["VALUE"].mean()
    .rename(columns={"VALUE": "Unemployment_QAvg_Pct"})
)

# ----------------------------
# Income quarterly (normalize Dollars + millions)
# Filters:
# - GEO != Canada
# - UOM == Dollars
# - SCALAR_FACTOR == millions
# - concept column == Disposable income (exact label)
# ----------------------------
income = numeric_value(income_raw)
income["Quarter"] = to_quarter_start_from_date(income["REF_DATE"])
income["Province"] = income["GEO"].astype(str).str.strip()

concept_col = None
for c in ["Income, consumption and saving","Income","Household sector transactions","Estimates"]:
    if c in income.columns:
        concept_col = c
        break
if concept_col is None:
    concept_col = find_col(income, "income") or find_col(income, "transactions")
if concept_col is None:
    raise ValueError(f"INCOME: concept column not found. Columns: {list(income.columns)}")

inc = income.loc[(income["GEO"] != "Canada") & (income["UOM"] == "Dollars")].copy()
inc = inc.loc[inc["SCALAR_FACTOR"].astype(str).str.lower() == "millions"].copy()

avail = set(inc[concept_col].dropna().unique())
label = next((x for x in DISPOSABLE_LABELS if x in avail), None)
if label is None:
    raise ValueError(f"INCOME: disposable label not found in '{concept_col}'. Sample: {sorted(list(avail))[:60]}")

inc = inc.loc[inc[concept_col] == label, ["Province","Quarter","VALUE","SCALAR_FACTOR"]]
inc = inc.rename(columns={"VALUE": "DisposableIncome_Q_MillionCAD"})
inc = apply_scalar(inc, "DisposableIncome_Q_MillionCAD").rename(columns={
    "DisposableIncome_Q_MillionCAD_CAD": "DisposableIncome_Q_CAD"
})

income_q = (
    inc.groupby(["Province","Quarter"], as_index=False)
    .agg({"DisposableIncome_Q_MillionCAD":"sum", "DisposableIncome_Q_CAD":"sum"})
)

# ----------------------------
# FINAL MERGE (no mortgages)
# ----------------------------
fact = (
    income_q
    .merge(cpi_all_q, on=["Province","Quarter"], how="left")
    .merge(cpi_shel_q, on=["Province","Quarter"], how="left")
    .merge(unemp_q, on=["Province","Quarter"], how="left")
    .merge(rates_q, on="Quarter", how="left")
    .sort_values(["Province","Quarter"])
    .reset_index(drop=True)
)

# ----------------------------
# OUTPUT
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
fact.to_csv(OUT_FILE, index=False)

print("\n✅ Wrote:", OUT_FILE)
print("Rows:", len(fact))
print("Most recent quarter:", fact["Quarter"].max())
print("\nNull rate (top columns):\n", fact.isna().mean().sort_values(ascending=False).head(10))
