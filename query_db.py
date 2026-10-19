"""
Meta-XAI Database Query CLI Tool
Allows examiners and developers to run real SQL queries against data/burnout_guard.db.
Usage:
  python query_db.py
  python query_db.py "SELECT student_id, priority_level, status FROM students LIMIT 5;"
  python query_db.py "SELECT * FROM intervention_logs;"
  python query_db.py --tables
"""
import sqlite3
import sys
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "burnout_guard.db"


def run_query(sql: str) -> None:
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        print(f"\n[Executing SQL]: {sql}\n" + "-" * 75)
        if HAS_PANDAS:
            df = pd.read_sql_query(sql, conn)
            if df.empty:
                print("(0 rows returned)")
            else:
                print(df.to_string(index=False))
        else:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description] if cursor.description else []
            print(" | ".join(headers))
            print("-" * 75)
            for r in rows:
                print(" | ".join(str(item) for item in r))
            print(f"\n({len(rows)} rows)")
    finally:
        conn.close()


def show_summary() -> None:
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [t[0] for t in cursor.fetchall()]

        print("=" * 75)
        print("  BURNOUT GUARD - SQLITE RELATIONAL DATABASE INSPECTOR")
        print(f"  Database File: {DB_PATH.name} ({round(DB_PATH.stat().st_size / 1024, 2)} KB)")
        print(f"  Tables Found:  {', '.join(tables)}")
        print("=" * 75)

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cursor.fetchone()[0]
            print(f"\n>>> Table: [{table}] ({cnt} rows)")
            if HAS_PANDAS:
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 4", conn)
                print(df.to_string(index=False))
            else:
                cursor.execute(f"SELECT * FROM {table} LIMIT 4")
                for r in cursor.fetchall():
                    print(r)

        print("\n" + "=" * 75)
        print("Tip: Pass any custom SQL query as an argument, e.g.:")
        print("  python query_db.py \"SELECT student_id, priority_level, status FROM students WHERE priority_level='P1' LIMIT 5;\"")
        print("=" * 75)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_arg = " ".join(sys.argv[1:])
        if query_arg in ("--tables", "-t", ".tables"):
            run_query("SELECT name, type FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        elif query_arg in ("--schema", "-s", ".schema"):
            run_query("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        else:
            run_query(query_arg)
    else:
        show_summary()
