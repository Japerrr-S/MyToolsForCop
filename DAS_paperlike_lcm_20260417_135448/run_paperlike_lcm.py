import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


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


def _run_one(reward: str, env: Dict[str, str]) -> Tuple[Path, Dict[str, str], Dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "DAS.py",
        "--config",
        f"config/paper_lcm.py:{reward}",
    ]

    print("\n=== Running:", " ".join(cmd), "===")
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

    rewards = ["aesthetic", "pick", "multi"]

    summary_rows = []
    for r in rewards:
        run_dir, scores, div = _run_one(r, env=env)
        row = {"reward": r, "run_dir": str(run_dir)}
        row.update(scores)
        row.update(div)
        summary_rows.append(row)

    out_path = Path("logs") / f"paperlike_lcm_table_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    all_keys = []
    for row in summary_rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nDone")
    print("Summary:", out_path)
    for row in summary_rows:
        print("-", row["reward"], "->", row["run_dir"])


if __name__ == "__main__":
    main()
