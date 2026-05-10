# DDPO 纵向对比实验报告

## 基线来源

基线采用 `Japerrr-S/MyToolsForCop` 中的正式 DDPO PyTorch 实验：

- 目录：`ddpo_pytorch_formal_2026-04-18_220318/ddpo_pytorch_formal_2026-04-18_220318`
- 配置：`config_formal_4090.py`
- 指标：`metrics.csv`、`epoch_metrics.csv`、`das_like_summary.csv`
- 任务：`imagenet_animals` prompt + `jpeg_compressibility` reward

为了便于直接拼接分析，新实验保持以下条件与基线一致：

- `seed = 42`
- `num_epochs = 20`
- `sample.batch_size = 1`
- `sample.num_batches_per_epoch = 4`
- `sample.num_steps = 50`
- `sample.eta = 1.0`
- `sample.guidance_scale = 5.0`
- `train.batch_size = 1`
- `train.gradient_accumulation_steps = 1`
- `train.num_inner_epochs = 1`
- `train.cfg = True`
- `train.adv_clip_max = 5`
- `train.max_grad_norm = 1.0`
- `pretrained.model = runwayml/stable-diffusion-v1-5`
- `use_lora = True`
- `prompt_fn = imagenet_animals`
- `reward_fn = jpeg_compressibility`
- `per_prompt_stat_tracking.buffer_size = 16`
- `per_prompt_stat_tracking.min_count = 16`

## 参数改动

| 实验 | 原参数值 | 新参数值 | 改动目的 |
| --- | --- | --- | --- |
| `lr_1e_4` | `train.learning_rate = 3e-4` | `train.learning_rate = 1e-4` | 降低策略更新步长，观察更保守更新是否能降低 PPO 训练震荡和 clip 触发比例。 |
| `lr_6e_4` | `train.learning_rate = 3e-4` | `train.learning_rate = 6e-4` | 提高策略更新步长，观察 DDPO 在相同采样预算下是否能更快提升 reward，以及是否带来 KL/clipfrac 上升。 |
| `clip_5e_4` | `train.clip_range = 1e-4` | `train.clip_range = 5e-4` | 放宽 PPO ratio 裁剪范围，观察在学习率不变时是否允许更充分的策略更新。 |

除上表列出的单项改动外，其余实验条件均与基线对齐。

## 输出文件

新实验输出在：

`/root/autodl-tmp/ddpo_pytorch_lr_sweep_2026-05-06`

主要文件：

- `metrics.csv`：逐条 wandb history 记录，包含 reward 行与训练 loss / approx_kl / clipfrac 行。
- `epoch_metrics.csv`：按 epoch 聚合后的指标，格式对齐原仓库 `epoch_metrics.csv`。
- `das_like_summary.csv`：每个 run 的末轮摘要，格式对齐原仓库 `das_like_summary.csv`，并额外保留 `run_id`、`learning_rate`、`clip_range` 便于分析。
- `comparison_summary.csv`：基线与 3 组新实验的关键结果汇总。
- `configs/`：三组实验配置。
- `stdout/`：三组训练日志。

## 结果概览

| 实验 | learning_rate | clip_range | best_epoch | best_reward_mean | final_reward_mean | final_approx_kl_mean | final_clipfrac_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_formal` | `3e-4` | `1e-4` | 8 | -72.33675 | -113.996 | 1.105e-09 | 0.025 |
| `lr_1e_4` | `1e-4` | `1e-4` | 8 | -69.4 | -123.74875 | 1.652e-10 | 0.010 |
| `lr_6e_4` | `6e-4` | `1e-4` | 8 | -69.92025 | -91.13675 | 5.301e-09 | 0.040 |
| `clip_5e_4` | `3e-4` | `5e-4` | 8 | -72.49725 | -123.588 | 7.770e-10 | 0.000 |

注：`jpeg_compressibility` reward 是 JPEG 文件大小的负值，因此数值越大（越接近 0）表示越好。

## 初步观察

`lr_1e_4` 在最佳 epoch 上略好于基线，但末轮退化更明显；它的末轮 KL 和 clipfrac 更低，说明更新更保守，但没有避免后期 reward 回落。

`lr_6e_4` 的最佳 epoch 与 `lr_1e_4` 接近，末轮表现明显好于基线和其他两组；同时末轮 approx_kl 和 clipfrac 上升，说明更大的学习率带来了更强的策略移动。在这次 20 epoch 小样本设置下，它是最值得继续扩展采样量或重复 seed 验证的方向。

`clip_5e_4` 没有改善末轮 reward，clipfrac 反而为 0，说明在当前设置中单独放宽 `clip_range` 并不是主要瓶颈；学习率对结果的影响更明显。

## 运行备注

本地环境的 `diffusers` 版本为 `0.32.2`，原 DDPO PyTorch 代码面向较旧的 `diffusers==0.17.1`。为了在当前环境完成实验，我只做了兼容性修补：

- `randn_tensor` 导入路径兼容新版 diffusers。
- LoRA 初始化改为当前 `LoraConfig + add_adapter` 接口。
- FP16 mixed precision 下将可训练 LoRA 参数保持为 FP32，避免 accelerate unscale FP16 gradients 报错。

这些修补不改变上述实验超参数和 DDPO 训练逻辑。
