import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _latest_subdir(path: Path) -> Path:
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run dirs found under: {path}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    return dict(zip(header, row))


def _run_one(tag: str, config_py: str, reward: str, env: Dict[str, str]) -> Tuple[Path, Dict[str, str], Dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "DAS.py",
        "--config",
        f"{config_py}:{reward}",
    ]

    print("\n=== Running:", tag, "===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    runs_root = Path("logs") / "PAPERLIKE_LCM" / reward
    run_dir = _latest_subdir(runs_root)

    eval_cmd = [sys.executable, "eval_folder.py", "--img_folder", str(run_dir)]
    print("=== Evaluating:", " ".join(eval_cmd), "===")
    subprocess.run(eval_cmd, check=True, env=env)

    score_csv = run_dir / "eval_results.csv"
    div_csv = run_dir / "eval_diversity_results.csv"
    scores = _read_single_row_csv(score_csv)
    div = _read_single_row_csv(div_csv)

    return run_dir, scores, div


def main() -> None:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")

    reward = "aesthetic"
    sweeps: List[Dict[str, str]] = [
        {
            "sweep": "gamma005",
            "config": "config/paper_lcm_gamma005.py",
            "changed_param": "smc.tempering_gamma",
            "old_value": "0.1",
            "new_value": "0.05",
        },
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

    summary_rows: List[Dict[str, str]] = []
    for sweep in sweeps:
        run_dir, scores, div = _run_one(sweep["sweep"], sweep["config"], reward=reward, env=env)
        row: Dict[str, str] = {
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
        summary_rows.append(row)

    out_dir = Path("logs") / "das_longitudinal_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"das_longitudinal_aesthetic_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    all_keys: List[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nDone")
    print("Summary:", out_path)
    for row in summary_rows:
        print("-", row["sweep"], row["reward"], "->", row["run_dir"])


if __name__ == "__main__":
    main()
