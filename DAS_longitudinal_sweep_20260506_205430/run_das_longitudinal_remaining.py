from run_das_longitudinal_sweep import main as _unused_main  # noqa: F401
from run_das_longitudinal_sweep import _run_one

import csv
import os
import time
from pathlib import Path
from typing import Dict, List


def _read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    return dict(zip(header, row))


def main() -> None:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")

    reward = "aesthetic"
    completed = {
        "sweep": "gamma005",
        "reward": reward,
        "config": "config/paper_lcm_gamma005.py",
        "changed_param": "smc.tempering_gamma",
        "old_value": "0.1",
        "new_value": "0.05",
        "run_dir": "logs/PAPERLIKE_LCM/aesthetic/2026.05.06_20.04.14",
    }
    completed.update(_read_single_row_csv(Path(completed["run_dir"]) / "eval_results.csv"))
    completed.update(_read_single_row_csv(Path(completed["run_dir"]) / "eval_diversity_results.csv"))

    remaining = [
        {
            "sweep": "gamma020",
            "config": "config/paper_lcm_gamma020.py",
            "changed_param": "smc.tempering_gamma",
            "old_value": "0.1",
            "new_value": "0.2",
        },
        {
            "sweep": "kl0025",
            "config": "config/paper_lcm_kl0025.py",
            "changed_param": "smc.kl_coeff",
            "old_value": "0.005",
            "new_value": "0.0025",
        },
    ]

    rows: List[Dict[str, str]] = [completed]
    for sweep in remaining:
        run_dir, scores, div = _run_one(sweep["sweep"], sweep["config"], reward=reward, env=env)
        row = {
            "sweep": sweep["sweep"],
            "reward": reward,
            "config": sweep["config"],
            "changed_param": sweep["changed_param"],
            "old_value": sweep["old_value"],
            "new_value": sweep["new_value"],
            "run_dir": str(run_dir),
        }
        row.update(scores)
        row.update(div)
        rows.append(row)

    out_dir = Path("logs") / "das_longitudinal_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"das_longitudinal_aesthetic_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    all_keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Done")
    print("Summary:", out_path)
    for row in rows:
        print("-", row["sweep"], "->", row["run_dir"])


if __name__ == "__main__":
    main()
