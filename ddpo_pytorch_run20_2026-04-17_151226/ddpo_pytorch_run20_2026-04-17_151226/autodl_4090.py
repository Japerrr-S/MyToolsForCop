import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # AutoDL/4090-friendly defaults:
    # - put all logs/checkpoints on the data disk (/root/autodl-tmp)
    # - keep LoRA enabled to fit comfortably in 24GB VRAM
    config.run_name = "jpeg_4090"
    config.seed = 42
    config.logdir = "/root/autodl-tmp/ddpo-pytorch-logs"

    # A shorter, but still end-to-end run for analysis.
    # (Sampling + reward + PPO updates + checkpoints + images logged via wandb)
    config.num_epochs = 20
    config.save_freq = 5
    config.num_checkpoint_limit = 5

    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50
    sample.eta = 1.0
    sample.guidance_scale = 5.0
    sample.batch_size = 1
    # More samples per epoch makes reward statistics less noisy.
    sample.num_batches_per_epoch = 4

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0

    ###### Prompt Function ######
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    config.reward_fn = "jpeg_compressibility"

    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    config.per_prompt_stat_tracking.buffer_size = 16
    config.per_prompt_stat_tracking.min_count = 16

    return config
