'''
Launch and wrap duckietown env.
'''

import gym
from gym_duckietown.simulator import Simulator #import simulator
from utils.wrappers.observation import *
from utils.wrappers.action import *
from utils.wrappers.reward import *
import random
from copy import deepcopy



def map_generator(one_map):
    #hand picked maps from auxiliary_tools.visualize_maps.maps_visualized
    if one_map is None:
        only_LF_maps = ["ETHZ_autolab_fast_track",
                        #"ETHZ_autolab_technical_track",
                        "ETHZ_autolab_technical_track_bordered",
                        #"ETHZ_loop",
                        "ETHZ_loop_bordered",
                        "MOOC_modcon",
                        #"Montreal_loop",
                        "experiment_loop",
                        "loop_empty",
                        #"small_loop",
                        "small_loop_bordered",
                        #"small_loop_cw",
                        #"zigzag_dists",
                        "zigzag_dists_bordered"
                        ]
        selected_map = random.choice(only_LF_maps)
    else:
        selected_map = one_map
    return selected_map



def launch_and_wrap_env(env_config, id=None, one_map=None):
    temp_config = deepcopy(env_config)
    env_config = temp_config["env"]
    wrapper_config = temp_config["wrapper"]
    black_out_config = temp_config["black_out"]

    try:
        #different ids for each worker
        id = env_config.worker_index
    except:
        id = 0

    if id is not None:

        # Add random map only if all workers rolled out,
        # i.e. in a training step all workers will be in the same map.
        #if (env_config.worker_index + 1) == default.ray_config["num_workers"]:
        #    env_config.update({"map_name" : map_generator()})


        env_config.update({"map_name" : map_generator(one_map)})
        env_config.update({"seed" : env_config["seed"]+id})

        #start simulator
        env = Simulator(**env_config)
    else:
        raise ValueError('No env id is generated for launching env.')


    #execute wrappers
    env = wrap_all(env, wrapper_config, black_out_config)

    return env


def wrap_all(env, wrapper_config: dict, black_out_config: dict):
    #env wrappers
    #env = 
    #to do:
    #   -add aido wrapper?


    #observation wrappers
    if black_out_config["enabled"]:
        env = BlackOutWrapper(
            env,
            black_out_config["every_n_th"],
            black_out_config["length"],
            black_out_config["visualize"]
            )

    env = ClipImageWrapper(env) #crop obs. image top
    env = ResizeWrapper(env) #resize obs.
    env = NormalizeWrapper(env)

    if wrapper_config["frame_stacking"]:
        env = ObservationBufferWrapper(env, obs_buffer_depth=wrapper_config["frame_stacking_depth"])
    #env = EncoderWrapper(env)
    
    #action wrappers
    env = Heading2WheelVelsWrapper(env, "heading")

    #reward wrappers
    env = DtRewardPosAngle(env)
    env = DtRewardVelocity(env)
    #caution!!!
    #env = DtRewardWheelDiff(env)
    env = DtRewardCollisionAvoidance(env)

    return env
