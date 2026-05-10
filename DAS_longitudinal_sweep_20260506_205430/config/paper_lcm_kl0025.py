import ml_collections

from config.general import general


def smc():
    config = general()
    config.project_name = "PAPERLIKE_LCM"

    config.smc = ml_collections.ConfigDict()
    config.smc.num_particles = 4
    config.smc.resample_strategy = "ssp"
    config.smc.ess_threshold = 0.5

    config.smc.tempering = "schedule"
    config.smc.tempering_schedule = "exp"
    config.smc.tempering_gamma = 0.1
    config.smc.tempering_start = 0

    config.smc.verbose = False

    config.pretrained.model = "SimianLuo/LCM_Dreamshaper_v7"
    config.mixed_precision = "fp16"

    config.sample.num_steps = 8
    config.sample.eta = 0.5
    config.sample.guidance_scale = 7.5
    config.sample.batch_size = 1

    config.max_vis_images = 32
    config.prompt_fn = "eval_hps_v2_all"

    return config


def aesthetic():
    config = smc()
    config.reward_fn = "aesthetic"
    config.smc.kl_coeff = 0.0025
    return config


def pick():
    config = smc()
    config.reward_fn = "pick"
    config.smc.kl_coeff = 0.00005
    return config


def multi():
    config = smc()
    config.reward_fn = "multi"
    config.aes_weight = 0.5
    config.smc.kl_coeff = 0.0025
    return config


def get_config(name):
    return globals()[name]()
