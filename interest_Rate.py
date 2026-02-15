import io
import os
import re
import requests
import pandas as pd

GROUP_CODE = "A4_RATES_MORTGAGES"
START_DATE = "2013-01-01"     # BoC series typically starts ~2013/2014
END_DATE = None

OUT_DIR = "output"
OUT_4COLS = os.path.join(OUT_DIR, "boc_mortgage_rates_4cols_simple.csv")
OUT_WIDE = os.path.join(OUT_DIR, "boc_A4_RATES_MORTGAGES_wide.csv")
OUT_SERIES_MAP = os.path.join(OUT_DIR, "boc_A4_RATES_MORTGAGES_series_map.csv")


def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\ufeff", "").replace("\n", " ").replace("\r", " ").strip()
    s = " ".join(s.split())
    s = s.replace('"', "")
    return s


def download_boc_group_csv_text(group_code: str, start_date: str, end_date: str | None) -> str:
    base = f"https://www.bankofcanada.ca/valet/observations/group/{group_code}/csv"
    params = {"start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.content.decode("utf-8", errors="replace")


def parse_series_and_observations(raw_text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines = raw_text.splitlines()

    # ---- SERIES block ----
    series_idx = next((i for i, ln in enumerate(lines) if clean_text(ln).strip() == "SERIES"), None)
    if series_idx is None:
        raise ValueError("Could not find SERIES section in response.")

    obs_marker_idx = next((i for i in range(series_idx + 1, len(lines)) if clean_text(lines[i]).strip() == "OBSERVATIONS"), None)
    if obs_marker_idx is None:
        raise ValueError("Could not find OBSERVATIONS marker after SERIES section.")

    series_block = "\n".join(lines[series_idx + 1 : obs_marker_idx]).strip()
    series_df = pd.read_csv(io.StringIO(series_block), engine="python")

    if not {"id", "label", "description"}.issubset(set(series_df.columns)):
        raise ValueError(f"SERIES table missing columns. Found: {series_df.columns.tolist()}")

    series_df["id"] = series_df["id"].map(clean_text)
    series_df["label"] = series_df["label"].map(clean_text)
    series_df["description"] = series_df["description"].map(clean_text)

    # ---- OBSERVATIONS table ----
    header_idx = next((i for i, ln in enumerate(lines) if re.match(r'^\s*"?date"?\s*,', ln.strip().lower())), None)
    if header_idx is None:
        print("DEBUG first 80 lines:\n", "\n".join(lines[:80]))
        raise ValueError("Could not find OBSERVATIONS header row (date,...).")

    obs_csv = "\n".join(lines[header_idx:])
    obs_df = pd.read_csv(io.StringIO(obs_csv), engine="python")
    obs_df.columns = [clean_text(c) for c in obs_df.columns]

    if "date" not in obs_df.columns:
        raise ValueError(f"OBSERVATIONS missing date column. Columns: {obs_df.columns.tolist()[:20]} ...")

    obs_df["date"] = pd.to_datetime(obs_df["date"], errors="coerce")
    obs_df = obs_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in obs_df.columns:
        if c != "date":
            obs_df[c] = pd.to_numeric(obs_df[c], errors="coerce")

    return series_df, obs_df


def find_series_id(series_df: pd.DataFrame, obs_df: pd.DataFrame, *, label: str, must_contain: list[str]) -> str:
    """
    Find series id where:
      - label matches (case-insensitive)
      - description contains ALL must_contain tokens (case-insensitive)
      - id exists in obs_df columns
    """
    s = series_df.copy()
    s["label_l"] = s["label"].str.lower()
    s["desc_l"] = s["description"].str.lower()

    label_l = label.lower()
    tokens = [t.lower() for t in must_contain]

    cand = s[s["label_l"].eq(label_l)].copy()
    for t in tokens:
        cand = cand[cand["desc_l"].str.contains(re.escape(t))]

    cand = cand[cand["id"].isin(obs_df.columns)].copy()

    if cand.empty:
        # Show closest mortgage-related options to help you quickly see what BoC calls them
        mortgage_like = s[s["desc_l"].str.contains("mortgage") | s["label_l"].str.contains("mortgage")][
            ["id", "label", "description"]
        ].head(40)
        raise ValueError(
            f"Could not find series for label='{label}' with tokens={must_contain}.\n\n"
            f"Here are sample mortgage-like series from this group (first 40):\n"
            f"{mortgage_like.to_string(index=False)}"
        )

    # Prefer “Total” if multiple
    cand["has_total"] = cand["desc_l"].str.contains("total")
    cand = cand.sort_values(["has_total"], ascending=False)

    return cand.iloc[0]["id"]


def build_4col_output(series_df: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build exactly the 4 requested columns using OUTSTANDING BALANCES (generic).
    """
    base_tokens = ["mortgages", "outstanding balances", "residential mortgages"]

    # Variable, insured
    var_ins = find_series_id(
        series_df, obs_df,
        label="Variable rate",
        must_contain=base_tokens + ["insured"]
    )

    # Fixed, 5y+
    fix_ins = find_series_id(
        series_df, obs_df,
        label="Fixed rate, 5 years and over",
        must_contain=base_tokens + ["insured"]
    )

    var_un = find_series_id(
        series_df, obs_df,
        label="Variable rate",
        must_contain=base_tokens + ["uninsured"]
    )

    fix_un = find_series_id(
        series_df, obs_df,
        label="Fixed rate, 5 years and over",
        must_contain=base_tokens + ["uninsured"]
    )

    out = obs_df[["date", var_ins, fix_ins, var_un, fix_un]].copy()
    out = out.rename(columns={
        "date": "Date",
        var_ins: "Mortgage_Variable_Insured_Pct",
        fix_ins: "Mortgage_Fixed_5YPlus_Insured_Pct",
        var_un:  "Mortgage_Variable_Uninsured_Pct",
        fix_un:  "Mortgage_Fixed_5YPlus_Uninsured_Pct",
    })
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading BoC group:", GROUP_CODE)
    print("Official endpoint used:")
    print(f"https://www.bankofcanada.ca/valet/observations/group/{GROUP_CODE}/csv?start_date={START_DATE}")

    raw = download_boc_group_csv_text(GROUP_CODE, START_DATE, END_DATE)
    series_df, obs_df = parse_series_and_observations(raw)

    # Always save these so you can inspect quickly
    series_df.to_csv(OUT_SERIES_MAP, index=False)
    obs_df.to_csv(OUT_WIDE, index=False)
    print("✅ Saved series map:", OUT_SERIES_MAP)
    print("✅ Saved wide obs:", OUT_WIDE)
    print("Wide shape:", obs_df.shape)

    # Now try to produce your 4 columns
    try:
        out4 = build_4col_output(series_df, obs_df)
        out4.to_csv(OUT_4COLS, index=False)
        print("✅ Saved final 4-column CSV:", OUT_4COLS)
        print("Date range:", out4["Date"].min(), "to", out4["Date"].max())
        print(out4.head(8))
    except Exception as e:
        print("\n❌ Could not build 4-column output due to matching rules.")
        print("Reason:\n", str(e))
        print("\nOpen this file and tell me which exact series you want:")
        print(" ->", OUT_SERIES_MAP)
