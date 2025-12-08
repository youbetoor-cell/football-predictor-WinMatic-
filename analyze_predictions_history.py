import sqlite3
import pandas as pd
from pathlib import Path

# --- 1. Locate the DB ---
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "predictions_history.db"

print(f"Using DB at: {DB_PATH}")
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at: {DB_PATH}")

# --- 2. Load the table into pandas ---
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM predictions_history;", conn)
conn.close()

print(f"\nTotal rows in predictions_history: {len(df)}")

# --- 3. Clean up labels (handle '', NULL, 'None') ---
for col in ["predicted_side", "actual_result"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"": None, "None": None, "null": None})
    )

eval_df = df.dropna(subset=["predicted_side", "actual_result"]).copy()
print(f"Rows with known prediction AND result: {len(eval_df)}")

if eval_df.empty:
    print("\nNo completed matches yet with results recorded.")
else:
    # --- 4. Basic accuracy metrics ---
    eval_df["correct"] = eval_df["predicted_side"] == eval_df["actual_result"]
    overall_acc = eval_df["correct"].mean()

    print(f"\nOverall accuracy: {overall_acc:.2%}")

    # Accuracy by predicted side (home/draw/away)
    by_side = (
        eval_df.groupby("predicted_side")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    print("\nAccuracy by predicted side:")
    print(by_side.to_string(index=False, formatters={"accuracy": "{:.2%}".format}))

    # Accuracy by league (if you ever add more)
    by_league = (
        eval_df.groupby("league")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    print("\nAccuracy by league:")
    print(by_league.to_string(index=False, formatters={"accuracy": "{:.2%}".format}))

    # --- 5. Optional: export evaluated rows to CSV for Excel ---
    out_path = BASE_DIR / "predictions_history_with_eval.csv"
    eval_df.to_csv(out_path, index=False)
    print(f"\nExported detailed data with correctness flag to: {out_path}")
