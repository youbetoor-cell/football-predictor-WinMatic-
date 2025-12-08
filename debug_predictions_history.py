import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "predictions_history.db"

print(f"Using DB at: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM predictions_history;", conn)
conn.close()

# Clean labels
for col in ["predicted_side", "actual_result"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"": None, "None": None, "null": None})
    )

eval_df = df.dropna(subset=["predicted_side", "actual_result"]).copy()
print(f"\nRows with prediction and result: {len(eval_df)}")

print("\nValue counts – predicted_side:")
print(eval_df["predicted_side"].value_counts(dropna=False))

print("\nValue counts – actual_result:")
print(eval_df["actual_result"].value_counts(dropna=False))

print("\nCross-tab of predicted vs actual:")
print(pd.crosstab(eval_df["predicted_side"], eval_df["actual_result"], margins=True))

# Show a few sample rows where model predicted away AND away actually won
sample = eval_df[
    (eval_df["predicted_side"] == "away") &
    (eval_df["actual_result"] == "away")
].head(10)

print("\nExamples where model predicted away AND away actually won:")
print(sample[[
    "fixture_id",
    "home_team",
    "away_team",
    "model_home_p",
    "model_draw_p",
    "model_away_p",
    "predicted_side",
    "actual_result",
]])

