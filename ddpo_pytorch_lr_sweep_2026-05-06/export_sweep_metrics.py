#!/usr/bin/env python3
import csv
import json
from pathlib import Path


RUNS = [
    {
        "run_id": "lr_1e_4",
        "wandb_dir": "offline-run-20260506_180810-hxf7uadi",
        "run_dir_glob": "jpeg_4090_lr_1e_4_*",
        "learning_rate": 1e-4,
        "clip_range": 1e-4,
    },
    {
        "run_id": "lr_6e_4",
        "wandb_dir": "offline-run-20260506_182100-39d4vtlt",
        "run_dir_glob": "jpeg_4090_lr_6e_4_*",
        "learning_rate": 6e-4,
        "clip_range": 1e-4,
    },
    {
        "run_id": "clip_5e_4",
        "wandb_dir": "offline-run-20260506_183346-vkyibfms",
        "run_dir_glob": "jpeg_4090_clip_5e_4_*",
        "learning_rate": 3e-4,
        "clip_range": 5e-4,
    },
]


def read_wandb_history(wandb_path: Path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

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

        row = {}
        history = rec.history
        try:
            if hasattr(history, "step") and hasattr(history.step, "num"):
                row["history_step_num"] = history.step.num
        except Exception:
            pass

        for item in history.item:
            key = "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                row[key] = item.value_json

        if "epoch" in row and len(row) > 1:
            rows.append(row)

    return rows


def mean(values):
    values = [float(v) for v in values if v != "" and v is not None]
    return sum(values) / len(values) if values else ""


def std(values):
    values = [float(v) for v in values if v != "" and v is not None]
    if not values:
        return ""
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def main():
    here = Path(__file__).resolve().parent
    wandb_root = here / "wandb_offline" / "wandb"
    run_root = here / "run"

    metrics_rows = []
    epoch_rows = []
    summary_rows = []

    for run in RUNS:
        wandb_files = sorted((wandb_root / run["wandb_dir"]).glob("*.wandb"))
        if not wandb_files:
            raise SystemExit(f"No wandb file found for {run['run_id']}")
        run_dirs = sorted(run_root.glob(run["run_dir_glob"]))
        if not run_dirs:
            raise SystemExit(f"No run directory found for {run['run_id']}")
        run_dir = str(run_dirs[-1])

        rows = read_wandb_history(wandb_files[-1])
        for row in rows:
            out = dict(row)
            out["run_id"] = run["run_id"]
            out["run_dir"] = run_dir
            out["reward_fn"] = "jpeg_compressibility"
            out["learning_rate"] = run["learning_rate"]
            out["clip_range"] = run["clip_range"]
            metrics_rows.append(out)

        epochs = sorted({int(row["epoch"]) for row in rows if "epoch" in row})
        for epoch in epochs:
            epoch_history = [row for row in rows if int(row.get("epoch", -1)) == epoch]
            train_history = [row for row in epoch_history if "loss" in row]
            reward_history = [row for row in epoch_history if "reward_mean" in row]
            epoch_row = {
                "run_id": run["run_id"],
                "run_dir": run_dir,
                "reward": "jpeg_compressibility",
                "learning_rate": run["learning_rate"],
                "clip_range": run["clip_range"],
                "epoch": epoch,
                "n": len(train_history),
                "loss_mean": mean(row.get("loss") for row in train_history),
                "approx_kl_mean": mean(row.get("approx_kl") for row in train_history),
                "clipfrac_mean": mean(row.get("clipfrac") for row in train_history),
                "reward_mean_mean": mean(row.get("reward_mean") for row in reward_history),
                "reward_std_mean": mean(row.get("reward_std") for row in reward_history),
                "loss_std": std(row.get("loss") for row in train_history),
                "approx_kl_std": std(row.get("approx_kl") for row in train_history),
                "clipfrac_std": std(row.get("clipfrac") for row in train_history),
                "reward_mean_std": std(row.get("reward_mean") for row in reward_history),
                "reward_std_std": std(row.get("reward_std") for row in reward_history),
            }
            epoch_rows.append(epoch_row)

        final = [row for row in epoch_rows if row["run_id"] == run["run_id"]][-1]
        summary_rows.append(
            {
                "reward": "jpeg_compressibility",
                "run_id": run["run_id"],
                "run_dir": run_dir,
                "learning_rate": run["learning_rate"],
                "clip_range": run["clip_range"],
                "epoch": final["epoch"],
                "n": final["n"],
                "reward_mean": final["reward_mean_mean"],
                "reward_std": final["reward_std_mean"],
                "loss_mean": final["loss_mean"],
                "loss_std": final["loss_std"],
                "approx_kl_mean": final["approx_kl_mean"],
                "approx_kl_std": final["approx_kl_std"],
                "clipfrac_mean": final["clipfrac_mean"],
                "clipfrac_std": final["clipfrac_std"],
            }
        )

    metric_cols = list(
        dict.fromkeys(
            [
                "run_id",
                "run_dir",
                "reward_fn",
                "learning_rate",
                "clip_range",
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
            + sorted({key for row in metrics_rows for key in row})
        )
    )
    epoch_cols = [
        "run_id",
        "run_dir",
        "reward",
        "learning_rate",
        "clip_range",
        "epoch",
        "n",
        "loss_mean",
        "approx_kl_mean",
        "clipfrac_mean",
        "reward_mean_mean",
        "reward_std_mean",
        "loss_std",
        "approx_kl_std",
        "clipfrac_std",
        "reward_mean_std",
        "reward_std_std",
    ]
    summary_cols = [
        "reward",
        "run_id",
        "run_dir",
        "learning_rate",
        "clip_range",
        "epoch",
        "n",
        "reward_mean",
        "reward_std",
        "loss_mean",
        "loss_std",
        "approx_kl_mean",
        "approx_kl_std",
        "clipfrac_mean",
        "clipfrac_std",
    ]

    for path, cols, data in [
        (here / "metrics.csv", metric_cols, metrics_rows),
        (here / "epoch_metrics.csv", epoch_cols, epoch_rows),
        (here / "das_like_summary.csv", summary_cols, summary_rows),
    ]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in data:
                writer.writerow({col: row.get(col, "") for col in cols})
        print(f"Wrote {path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
