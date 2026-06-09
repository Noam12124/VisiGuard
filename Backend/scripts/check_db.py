"""Quick SQLite gallery integrity check."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

db_path = Path(__file__).resolve().parent.parent / "data" / "visiguard.db"
if not db_path.exists():
    print("DB not found:", db_path)
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
print("Tables:", cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print("Users:", cur.execute("SELECT id, username FROM users").fetchall())

for eid, uid, name in cur.execute("SELECT id, user_id, person_name FROM gallery"):
    try:
        emb_raw = cur.execute("SELECT embedding FROM gallery WHERE id=?", (eid,)).fetchone()[0]
        emb = json.loads(emb_raw)
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        print(
            f"Gallery id={eid} user={uid} name={name!r} "
            f"dims={arr.shape} nulls={any(x is None for x in emb)} "
            f"min={arr.min():.4f} max={arr.max():.4f}"
        )
    except Exception as exc:
        print(f"Gallery id={eid} CORRUPT: {exc}")

print("Recognition events:", cur.execute("SELECT COUNT(*) FROM recognition_events").fetchone()[0])
print("Unknown persons:", cur.execute("SELECT COUNT(*) FROM unknown_persons").fetchone()[0])
conn.close()
