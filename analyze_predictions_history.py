#!/usr/bin/env python3
import sqlite3
from pathlib import Path
from textwrap import dedent

DB_PATH = Path("data/predictions_history.db")


def main():
    db_path = DB_PATH.resolve()
    print(f"Using DB at: {db_path}\n")

    if not db_path.exists():
        print("ERROR: DB does not exist.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 1) Basic counts
    cur.execute("SELECT COUNT(*) FROM predictions_history")
    total_rows = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        """
    )
    known_rows = cur.fetchone()[0]

    print(f"Total rows in predictions_history: {total_rows}")
    print(f"Rows with known prediction AND result: {known_rows}\n")

    if known_rows == 0:
        print("No rows with both prediction and result. Nothing to analyze.")
        conn.close()
        return

    # 2) Distinct predicted_side and actual_result
    print("Distinct predicted_side values in known rows:")
    cur.execute(
        """
        SELECT DISTINCT predicted_side
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        ORDER BY predicted_side
        """
    )
    for (ps,) in cur.fetchall():
        print(f"  '{ps}'")
    print()

    print("Distinct actual_result values in known rows:")
    cur.execute(
        """
        SELECT DISTINCT actual_result
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        ORDER BY actual_result
        """
    )
    for (ar,) in cur.fetchall():
        print(f"  '{ar}'")
    print()

    # 3) Overall accuracy
    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
          AND predicted_side = actual_result
        """
    )
    correct = cur.fetchone()[0]
    overall_acc = correct / known_rows * 100.0
    print(f"Overall accuracy: {overall_acc:.2f}%\n")

    # 4) Accuracy by predicted_side
    print("Accuracy by predicted side:")
    print("predicted_side  accuracy     n")

    cur.execute(
        """
        SELECT predicted_side,
               SUM(CASE WHEN predicted_side = actual_result THEN 1 ELSE 0 END) AS correct,
               COUNT(*) AS n
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        GROUP BY predicted_side
        ORDER BY predicted_side
        """
    )
    for side, correct_side, n_side in cur.fetchall():
        acc_side = (correct_side / n_side * 100.0) if n_side else 0.0
        print(f"{side:>13}  {acc_side:7.2f}% {n_side:5d}")
    print()

    # 5) Accuracy by league
    print("Accuracy by league:")
    print("  league  accuracy     n")
    cur.execute(
        """
        SELECT league,
               SUM(CASE WHEN predicted_side = actual_result THEN 1 ELSE 0 END) AS correct,
               COUNT(*) AS n
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        GROUP BY league
        ORDER BY league
        """
    )
    for league, correct_lg, n_lg in cur.fetchall():
        acc_lg = (correct_lg / n_lg * 100.0) if n_lg else 0.0
        print(f"{league:8}  {acc_lg:7.2f}% {n_lg:5d}")
    print()

    # 6) Simple confusion-style table: predicted_side x actual_result
    print("Confusion-style counts (predicted_side x actual_result):")
    cur.execute(
        """
        SELECT predicted_side,
               actual_result,
               COUNT(*) as n
        FROM predictions_history
        WHERE predicted_side IS NOT NULL
          AND predicted_side != ''
          AND actual_result IN ('home', 'away', 'draw')
        GROUP BY predicted_side, actual_result
        ORDER BY predicted_side, actual_result
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("  (no data)")
    else:
        # Format nicely
        print("  predicted  actual  count")
        for ps, ar, n in rows:
            print(f"  {ps:8}  {ar:5}  {n:5d}")
    print()

    conn.close()

    # 7) Reminder about where the CSV lives (unchanged)
    out_csv = Path("predictions_history_with_eval.csv").resolve()
    print(f"Exported detailed data with correctness flag to: {out_csv}")


if __name__ == "__main__":
    main()

