# How to read this DDPO archive

This archive follows the same nested-directory convention as the DDPO data in MyToolsForCop.

- `metrics.csv`: raw exported wandb history rows for the three successful runs.
- `epoch_metrics.csv`: epoch-level aggregation aligned with the previous DDPO archive.
- `das_like_summary.csv`: final-epoch summary table aligned with the previous DDPO summary style.
- `comparison_summary.csv`: baseline + new runs in one table for direct analysis.
- `parameter_change_report.md`: parameter meaning, old/new values, impact, and conclusions.
- `configs/`: exact configs used for each sweep run.
- `run/`: checkpoint directories produced by successful training runs.
- `stdout/`: terminal logs for each successful run.
- `wandb_offline/`: completed offline wandb run records only.

Note: this DDPO run did not produce standalone `.png`/`.jpg` media files on disk. The archive keeps the successful
offline wandb records and checkpoints so images/curves can be regenerated or inspected from the run records if needed.
