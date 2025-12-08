import sqlite3
import pandas as pd
from pathlib import Path

# Resolve DB path relative to THIS file's directory
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "predictions_history.db"

print(f"Using DB at: {DB_PATH}")

# Sanity check: does it exist and have size?
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at: {DB_PATH}")

print(f"DB size: {DB_PATH.stat().st_size} bytes")

# Connect to the database
conn = sqlite3.connect(DB_PATH)

# List tables
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)
print("\nTables found:\n", tables)

# Try to read predictions_history
try:
    df = pd.read_sql_query("SELECT * FROM predictions_history;", conn)
    print("\nFirst 10 rows of predictions_history:\n")
    print(df.head(10))
except Exception as e:
    print("\nCould not read predictions_history table:", e)
    df = None

# Optional CSV export
if df is not None:
    out_path = BASE_DIR / "predictions_history_export.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExported to {out_path}")

conn.close()
