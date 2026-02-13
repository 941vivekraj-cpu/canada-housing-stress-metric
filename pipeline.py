import os
import json
import pandas as pd
import duckdb

def main():
    # Always create folders (fixes your earlier error)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    db_path = "data/caneco.duckdb"
    con = duckdb.connect(db_path)

    # -------------------------
    # TODO: Replace this block with your real pipeline
    # For now, create a tiny sample fact table so GitHub Actions proves it's working.
    # -------------------------
    fact = pd.DataFrame({
        "Province": ["Ontario", "Ontario", "Alberta", "Alberta"],
        "Quarter": ["2025Q3", "2025Q4", "2025Q3", "2025Q4"],
        "CPI_QAvg": [160.2, 161.1, 158.9, 159.4],
        "Unemployment_QAvg": [6.1, 6.4, 5.8, 6.0],
        "DisposableIncome_Q_Millions": [100000, 101500, 70000, 70500],
        "PrimeRate_QEnd": [6.95, 6.95, 6.95, 6.95],
    })

    con.register("fact_df", fact)
    con.execute("CREATE OR REPLACE TABLE fact_household_stress AS SELECT * FROM fact_df")

    # Export CSV for Tableau
    out_csv = "output/fact_household_stress.csv"
    con.execute(f"""
        COPY fact_household_stress
        TO '{out_csv}'
        WITH (HEADER, DELIMITER ',');
    """)

    # Basic QA + run log (senior signal)
    run_log = {
        "db_path": db_path,
        "rows_fact": int(len(fact)),
        "dupes_province_quarter": int(fact.duplicated(["Province", "Quarter"]).sum()),
        "null_rate_percent": float(fact.isna().mean().max() * 100),
        "columns": list(fact.columns),
    }
    with open("output/run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    con.close()
    print("✅ Wrote:", out_csv)
    print("✅ Wrote: output/run_log.json")
    print("Run log:", run_log)

if __name__ == "__main__":
    main()
