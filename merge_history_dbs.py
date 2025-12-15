import sqlite3
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_history_dbs.py TARGET_DB SOURCE1.db [SOURCE2.db ...]")
        sys.exit(1)

    target_db = sys.argv[1]
    sources = sys.argv[2:]

    if not os.path.exists(target_db):
        print("Target DB %s does not exist. Start your app once so it creates it." % target_db)
        sys.exit(1)

    print("Target DB:", target_db)
    print("Sources:", ", ".join(sources))

    # Open target connection
    t_conn = sqlite3.connect(target_db)
    t_cur = t_conn.cursor()

    for src in sources:
        if not os.path.exists(src):
            print("Skipping %s (file not found)" % src)
            continue

        print("---- Merging %s ----" % src)
        s_conn = sqlite3.connect(src)
        s_cur = s_conn.cursor()

        try:
            # Get column list from source table
            s_cur.execute("PRAGMA table_info(predictions_history)")
            cols_info = s_cur.fetchall()
            if not cols_info:
                print("  No table 'predictions_history' in %s, skipping." % src)
                s_conn.close()
                continue

            # Use all columns except the primary key 'id'
            cols = [row[1] for row in cols_info if row[1] != "id"]
            cols_sql = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))

            # Read all rows from source
            s_cur.execute("SELECT %s FROM predictions_history" % cols_sql)
            rows = s_cur.fetchall()
            if not rows:
                print("  No rows to merge from %s." % src)
                s_conn.close()
                continue

            insert_sql = "INSERT INTO predictions_history (%s) VALUES (%s)" % (
                cols_sql,
                placeholders,
            )

            added = 0
            for row in rows:
                t_cur.execute(insert_sql, row)
                added += 1

            t_conn.commit()
            print("  Inserted %d rows from %s" % (added, src))

        except Exception as e:
            print("  ERROR merging %s: %s" % (src, e))
        finally:
            s_conn.close()

    t_conn.close()
    print("Done merging.")


if __name__ == "__main__":
    main()

