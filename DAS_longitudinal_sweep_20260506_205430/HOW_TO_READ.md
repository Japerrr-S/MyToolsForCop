# How to read this DAS archive

This archive mirrors the DAS paper-like LCM layout in MyToolsForCop.

- `config/`: baseline-aligned configs plus the new KL sweep config.
- `logs/PAPERLIKE_LCM/aesthetic/<run>/config.json`: resolved runtime config for each successful run.
- `logs/PAPERLIKE_LCM/aesthetic/<run>/eval_results.csv`: reward/evaluator metrics with the same columns as the original archive.
- `logs/PAPERLIKE_LCM/aesthetic/<run>/eval_diversity_results.csv`: diversity metrics with the same columns as the original archive.
- `logs/PAPERLIKE_LCM/aesthetic/<run>/eval_vis/`: generated images and diagnostic traces for thesis figures.
- `logs/paperlike_lcm_table_20260506_205430.csv`: original-column summary table for direct compatibility.
- `logs/das_longitudinal_sweep/das_longitudinal_aesthetic_20260506_205430.csv`: normalized sweep table with parameter-change columns.
- `logs/das_longitudinal_sweep/das_aesthetic_comparison_summary.csv`: baseline + new runs comparison.
- `logs/das_longitudinal_sweep/das_parameter_change_report.md`: parameter meaning, old/new values, impact, and conclusions.
