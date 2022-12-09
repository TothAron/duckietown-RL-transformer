"""
Script to train the transformer agent.
"""

#general
import random
import numpy as np


#environment
from utils.env import launch_and_wrap_env
from ray.tune.registry import register_env

#logging
from utils import saver
from utils.extra_tools import show_model_statistics
from utils.callbacks import on_episode_start, on_episode_step, on_episode_end, on_train_result
from utils.logger import WeightsAndBiasesLogger

#import config
from config import default

# checkponting
from utils.chkpt_utils import config_and_chkpt_initalizer

#framework
import ray
from ray import tune
import ray.rllib
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from model.wrapped_cnn import Custom_wrapped_cnn

#hyperopt
from utils.wandb_sweep_parser import *


#set seed
np.random.seed(1234)
random.seed(1234)


###########
if __name__ == "__main__":
    # checkpoint initalizer
    default, ray_config, chkpt_config, chkpt_path = config_and_chkpt_initalizer(default)

    #initialize ray runtime for paralellization
    ray.shutdown()
    ray.init(**default.ray_init_config)

    # register duckietown env
    register_env("Duckietown", launch_and_wrap_env)
    
    #import custom model
    ModelCatalog.register_custom_model("custom_wrapped_cnn", Custom_wrapped_cnn) 


    config = {
        "env": "Duckietown",  # define env
        "env_config": {
            "env": default.env_config,
            "wrapper": default.wrapper_config,
            "black_out": default.black_out_config,
            },
        "callbacks": {
            'on_episode_start': on_episode_start,
            'on_episode_step': on_episode_step,
            'on_episode_end': on_episode_end,
            'on_train_result': on_train_result,
            'wandb_config_pass': default.wandb_config,
        }
    }

    config.update(ray_config) # add/concatenate ray configs
    config, plain_wandb_config = wandb_sweep_hps_override(config) #modify hps with wandb sweep controlled hps
    stop = default.ray_stop_config # import ray stop config

    print("Running with the following config:\n", config)

    ## get model parameters and architecture
    #show_model_statistics(config)


    train_name = saver.save_scripts(plain_wandb_config)
    # Training automatically with Ray Tune
    analysis = tune.run(
        ppo.PPOTrainer,
        config=config,
        name =train_name,
        verbose=2,
        stop=stop,
        local_dir="./logs",
        loggers=[WeightsAndBiasesLogger],
        log_to_file="output.log",
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=1,
        checkpoint_freq=1,
        restore=chkpt_path,
        )

    ray.shutdown()
