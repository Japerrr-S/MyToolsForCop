#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Iterable


def iter_wandb_files(wandb_dir: Path) -> Iterable[Path]:
    if not wandb_dir.exists():
        return []
    return sorted(wandb_dir.rglob("*.wandb"), key=lambda p: p.stat().st_mtime)


def main() -> None:
    here = Path(__file__).resolve().parent
    wandb_dir = here / "wandb_offline"
    out_csv = here / "metrics.csv"

    wandb_files = list(iter_wandb_files(wandb_dir))
    if not wandb_files:
        raise SystemExit(f"No .wandb files found under: {wandb_dir}")

    wandb_path = wandb_files[-1]

    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2

    rows = []
    ds = DataStore()
    ds.open_for_scan(str(wandb_path))

    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if not rec.HasField("history"):
            continue
        h = rec.history
        row = {}

        try:
            if hasattr(h, "step") and hasattr(h.step, "num"):
                row["history_step_num"] = h.step.num
        except Exception:
            pass

        for it in h.item:
            key = "/".join(it.nested_key)
            vraw = it.value_json
            try:
                val = json.loads(vraw)
            except Exception:
                val = vraw
            row[key] = val

        # Keep rows that look like training history (epoch present and at least one metric)
        if "epoch" in row and len(row) > 1:
            rows.append(row)

    core_cols = [
        "epoch",
        "inner_epoch",
        "_step",
        "history_step_num",
        "loss",
        "approx_kl",
        "clipfrac",
        "reward",
        "reward_mean",
        "reward_std",
        "_runtime",
        "_timestamp",
    ]
    all_cols = list(dict.fromkeys(core_cols + sorted({k for r in rows for k in r.keys()})))

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_cols})

    print(f"Using: {wandb_path}")
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(all_cols)}")


if __name__ == "__main__":
    main()
