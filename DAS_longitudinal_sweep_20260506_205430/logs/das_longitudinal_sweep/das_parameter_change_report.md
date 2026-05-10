# DAS 纵向对比实验报告

## 基线来源

基线采用 `Japerrr-S/MyToolsForCop` 中的 DAS paper-like LCM 实验：

- 仓库目录：`DAS_paperlike_lcm_20260417_135448`
- 基线配置：`config/paper_lcm.py:aesthetic`
- 基线结果：`logs/paperlike_lcm_table_20260417_135448.csv`
- 具体 run：`logs/PAPERLIKE_LCM/aesthetic/2026.04.17_13.28.04`

本次纵向对比固定任务为 `aesthetic`，因为用户要求比较 DAS 方法自身，而不是不同 reward 任务之间的差异。

## 对齐参数

三组新实验均保持以下配置与基线一致：

- `seed = 42`
- `pretrained.model = SimianLuo/LCM_Dreamshaper_v7`
- `mixed_precision = fp16`
- `prompt_fn = eval_hps_v2_all`
- `reward_fn = aesthetic`
- `max_vis_images = 32`
- `sample.batch_size = 1`
- `sample.num_steps = 8`
- `sample.eta = 0.5`
- `sample.guidance_scale = 7.5`
- `smc.num_particles = 4`
- `smc.resample_strategy = ssp`
- `smc.ess_threshold = 0.5`
- `smc.tempering = schedule`
- `smc.tempering_schedule = exp`
- `smc.tempering_start = 0`

## 参数改动

| 实验 | 原参数值 | 新参数值 | 改动原因 |
| --- | --- | --- | --- |
| `gamma005` | `smc.tempering_gamma = 0.1` | `smc.tempering_gamma = 0.05` | 降低 tempering schedule 强度，让奖励引导更平缓，观察是否能保留更多样性并降低采样过程的过强偏置。 |
| `gamma020` | `smc.tempering_gamma = 0.1` | `smc.tempering_gamma = 0.2` | 提高 tempering schedule 强度，让奖励影响更快进入 SMC 权重，观察目标 reward 是否能提升。 |
| `kl0025` | `smc.kl_coeff = 0.005` | `smc.kl_coeff = 0.0025` | 降低 KL 约束强度，允许粒子更偏向 reward 高的区域，观察 aesthetic reward 与多样性的权衡。 |

DAS 是采样时优化/重加权方法，不包含 DDPO 那类训练学习率；因此这里选择 DAS/SMC 中最直接影响纵向行为的 `tempering_gamma` 与 `kl_coeff`。

## 输出文件

新实验和报告位于：

`/root/autodl-tmp/DAS/logs/das_longitudinal_sweep`

主要文件：

- `das_longitudinal_aesthetic_20260506_205430.csv`：三组新实验的汇总，列格式对齐原仓库 `paperlike_lcm_table_20260417_135448.csv`，并额外加入参数变更列。
- `das_aesthetic_comparison_summary.csv`：基线 + 三组新实验的统一对比表。
- `das_parameter_change_report.md`：本报告。

各 run 的原始输出：

- `gamma005`：`logs/PAPERLIKE_LCM/aesthetic/2026.05.06_20.04.14`
- `gamma020`：`logs/PAPERLIKE_LCM/aesthetic/2026.05.06_20.38.13`
- `kl0025`：`logs/PAPERLIKE_LCM/aesthetic/2026.05.06_20.46.25`

## 结果概览

| 实验 | tempering_gamma | kl_coeff | aesthetic_mean | imagereward_mean | pick_mean | clip_mean | mean_pairwise_distance_clip | mean_lpips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_formal` | 0.1 | 0.005 | 6.255701 | 0.648130 | 0.223326 | 0.252887 | 0.466897 | 0.527390 |
| `gamma005` | 0.05 | 0.005 | 6.057562 | 0.381036 | 0.223964 | 0.252888 | 0.464587 | 0.520593 |
| `gamma020` | 0.2 | 0.005 | 6.392585 | 0.286562 | 0.221657 | 0.249080 | 0.454532 | 0.560327 |
| `kl0025` | 0.1 | 0.0025 | 6.499646 | 0.292491 | 0.218636 | 0.241389 | 0.437325 | 0.556814 |

## 初步观察

`gamma005` 相比基线没有提升目标 aesthetic score，说明更弱的 tempering 在当前 32 图、8 step LCM 设置下对目标 reward 的推动不足；多样性指标也没有明显优势。

`gamma020` 提高了 `aesthetic_mean`，说明增强 tempering 能更积极地把粒子推向 aesthetic scorer 偏好的区域；但 `clip_mean` 和 CLIP pairwise diversity 有所下降，表示文本一致性和 embedding 多样性存在一定代价。

`kl0025` 的 `aesthetic_mean` 最高，但 `pick_mean`、`clip_mean` 和 `mean_pairwise_distance_clip` 最低，说明降低 KL 约束确实增强了 reward 优化力度，同时也更明显牺牲了多样性和部分跨指标表现。

后续如果要扩展，建议优先在 `gamma020` 与 `kl0025` 附近做更细粒度搜索，例如 `tempering_gamma = 0.15` 或 `kl_coeff = 0.0035`，并增加 seed 重复来区分真实趋势与 32 张样本下的方差。

## 运行备注

第一次评估时本地缺少 `ImageReward` 包；已从 `THUDM/ImageReward` 安装，并使用 `HF_ENDPOINT=https://hf-mirror.com` 下载权重后完成评估。最终输出表包含与原仓库一致的 `imagereward_mean/std` 列。
