from __future__ import annotations

from pathlib import Path
import csv

from .memory import MemoryStore


def export_research_bundle(memory: MemoryStore, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generations = memory.conn.execute("SELECT * FROM generations ORDER BY id").fetchall()
    gpath = out / "generations.csv"
    with gpath.open("w", newline="", encoding="utf-8") as f:
        if generations:
            w = csv.DictWriter(f, fieldnames=generations[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in generations)

    metrics = memory.conn.execute("SELECT * FROM metrics ORDER BY id").fetchall()
    mpath = out / "metrics.csv"
    with mpath.open("w", newline="", encoding="utf-8") as f:
        if metrics:
            w = csv.DictWriter(f, fieldnames=metrics[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in metrics)

    insights = memory.conn.execute("SELECT * FROM insights ORDER BY id").fetchall()
    ipath = out / "insights.csv"
    with ipath.open("w", newline="", encoding="utf-8") as f:
        if insights:
            w = csv.DictWriter(f, fieldnames=insights[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in insights)

    games = memory.conn.execute("SELECT * FROM games ORDER BY created_at").fetchall()
    ppath = out / "games.pgn"
    with ppath.open("w", encoding="utf-8") as f:
        for g in games:
            f.write(g["pgn"].rstrip() + "\n\n")

    return {
        "generations": str(gpath),
        "metrics": str(mpath),
        "insights": str(ipath),
        "games": str(ppath),
    }
