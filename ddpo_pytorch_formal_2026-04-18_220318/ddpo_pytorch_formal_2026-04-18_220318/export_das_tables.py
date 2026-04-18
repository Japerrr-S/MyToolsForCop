#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


NUM_FIELDS = [
    "loss",
    "approx_kl",
    "clipfrac",
    "reward_mean",
    "reward_std",
]


def to_float(x: str):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    here = Path(__file__).resolve().parent
    metrics_path = here / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}. Run export_metrics.py first.")

    rows = []
    with metrics_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # --- epoch_metrics.csv (one row per epoch) ---
    by_epoch = defaultdict(list)
    for row in rows:
        e = row.get("epoch", "")
        try:
            epoch = int(float(e))
        except Exception:
            continue
        by_epoch[epoch].append(row)

    epoch_out = here / "epoch_metrics.csv"
    with epoch_out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "n",
                *[f"{k}_mean" for k in NUM_FIELDS],
                *[f"{k}_std" for k in NUM_FIELDS],
            ],
        )
        w.writeheader()
        for epoch in sorted(by_epoch.keys()):
            group = by_epoch[epoch]
            out = {"epoch": epoch, "n": len(group)}
            for k in NUM_FIELDS:
                vals = [to_float(r.get(k, "")) for r in group]
                vals = [v for v in vals if v is not None]
                if vals:
                    out[f"{k}_mean"] = mean(vals)
                    out[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
                else:
                    out[f"{k}_mean"] = ""
                    out[f"{k}_std"] = ""
            w.writerow(out)

    # --- das_like_summary.csv (single row, DAS-style columns) ---
    # Match DAS convention: (reward, run_dir, <metric>_mean, <metric>_std, ...)
    # Here: reward=reward_fn, run_dir points to the accelerate project dir (logdir/run_name).
    # We use the LAST epoch's aggregated stats as the most representative endpoint.
    last_epoch = max(by_epoch.keys()) if by_epoch else None

    # Parse reward_fn and run_dir from artifacts (best-effort).
    reward_fn = "jpeg_compressibility"
    run_dir = ""
    cfg_path = here / "config_formal_4090.py"
    if cfg_path.exists():
        txt = cfg_path.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            if "config.reward_fn" in line and "=" in line:
                reward_fn = line.split("=", 1)[1].strip().strip('"').strip("'")

    # Prefer the newest accelerate output directory under ./run
    run_root = here / "run"
    if run_root.exists():
        subdirs = [p for p in run_root.iterdir() if p.is_dir()]
        if subdirs:
            newest = sorted(subdirs, key=lambda p: p.stat().st_mtime)[-1]
            run_dir = str(newest)
    if not run_dir:
        run_dir = str(run_root) if run_root.exists() else "run"

    summary_out = here / "das_like_summary.csv"
    with summary_out.open("w", newline="") as f:
        fieldnames = [
            "reward",
            "run_dir",
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        if last_epoch is None:
            w.writerow({"reward": reward_fn, "run_dir": run_dir})
        else:
            # reuse epoch_out computation by recomputing for last epoch
            group = by_epoch[last_epoch]
            def agg(k):
                vals = [to_float(r.get(k, "")) for r in group]
                vals = [v for v in vals if v is not None]
                if not vals:
                    return ("", "")
                return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)

            rm, rs = agg("reward_mean")
            lm, ls = agg("loss")
            km, ks = agg("approx_kl")
            cm, cs = agg("clipfrac")

            w.writerow(
                {
                    "reward": reward_fn,
                    "run_dir": run_dir,
                    "epoch": last_epoch,
                    "n": len(group),
                    "reward_mean": rm,
                    "reward_std": rs,
                    "loss_mean": lm,
                    "loss_std": ls,
                    "approx_kl_mean": km,
                    "approx_kl_std": ks,
                    "clipfrac_mean": cm,
                    "clipfrac_std": cs,
                }
            )

    print(f"Wrote: {epoch_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
