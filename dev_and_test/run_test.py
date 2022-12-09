"""
Script to test the model.
"""

# checkponting
from utils.chkpt_utils import model_initalizer
from config.test_config import test_config, best_models, black_out_config

# RL model
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from utils.env import launch_and_wrap_env
from ray.rllib.models import ModelCatalog
from model.wrapped_cnn import Custom_wrapped_cnn
from utils.trajectory_plot import plot_trajectories, correct_gym_duckietown_coordinates
from utils.duckietown_evaluator import DuckietownWorldEvaluator
from gym_duckietown.simulator import DEFAULT_ROBOT_SPEED

from utils.extra_tools import show_model_statistics

# rendering
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb

class EvalRes():
    def __init__(
        self,
        rewards=[],
        reward_return=0.0,
        actions=[],
        positions=[],
        map_img=None,
        plot=None
        ):
        self.rewards = rewards
        self.reward_return = reward_return
        self.actions = actions
        self.positions = positions
        self.map_img = map_img
        self.plot = plot

def init_logging(test_config, model_type, map):
    timestamp = str(datetime.now())
    wandb_config = test_config.wandb_config
    wandb_config["group"] = "EVAL"
    wandb_config["name"] =  f"EVAL_{model_type}_{timestamp}_{map}"
    wandb_logger =  wandb.init(**wandb_config)
    eval_res = EvalRes([],0.0,[],[],None,None)
    return wandb_logger

def log_on_step(eval_res, action, reward, env, info):
    eval_res.actions.append(action)
    eval_res.rewards.append(reward)
    position= correct_gym_duckietown_coordinates(
            env.unwrapped,
            env.unwrapped.cur_pos)
    eval_res.positions.append(position)
    
    wandb.log({
        'actions': action,
        'rewards': reward,
        'return': np.sum(eval_res.rewards),
        'position': position 
    },
    step=env.unwrapped.step_count)
    return eval_res

def initalize_env(test_config: dict):
    env = None
    env_config = {
        "env": test_config.env_config,
        "wrapper": test_config.wrapper_config
    }
    env = launch_and_wrap_env(
        env_config,
        one_map = test_config.env_config["map_name"]
        )
    obs = env.reset()
    env.render("human")
    done = False
    return env, obs, done

def init_RL_variables(model_type: str, test_config: dict, ray_config: dict):
    if model_type == "CNN_TR":
        num_transformers = ray_config["model"]["attention_num_transformer_units"]
        attention_dim = ray_config["model"]["attention_dim"]
        memory = ray_config["model"]["attention_memory_inference"]
        state = [np.zeros([memory, attention_dim], np.float32) for _ in range(num_transformers)]
        action = np.zeros(ray_config["model"]["attention_use_n_prev_actions"],)
        reward = np.zeros(ray_config["model"]["attention_use_n_prev_rewards"],)
        seq_len = num_transformers
    elif model_type in ["CNN", "CNN_framestacking"]:
        state = np.zeros(1)
        action = np.zeros(1)
        reward = np.zeros(1)
        seq_len = 0
    elif model_type == "CNN_LSTM":
        seq_len = ray_config["model"]["max_seq_len"]
        cell_size = ray_config["model"]["lstm_cell_size"]
        state = [np.zeros([cell_size], np.float32) for _ in range(2)]
        action = np.zeros(1)
        reward = np.zeros(1)
    else:
        raise f"No implementation for {model_type} model type."

    return state, action, reward, seq_len

def run_evaluation(model_type, trainer, env, obs, done, test_config, ray_config):
    state, action, reward, seq_len = init_RL_variables(model_type, test_config, ray_config)
    out = cv2.VideoWriter(f"evaluation_results/{model_type}.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    eval_res = init_logging(test_config, model_type)
    #RL loop
    while not done:
        action_out, state_out, _ = trainer.compute_single_action(
            observation=obs,
            prev_action=action,
            prev_reward = reward,
            state=state,
            full_fetch=True,
            explore=False,
            policy_id="default_policy")
        obs, reward_out, done, info = env.step(action_out)

        if model_type in "CNN_TR":
            state = [np.concatenate([state[i], [state_out[i]]], axis=0)[1:] for i in range(seq_len)]
            action = np.concatenate([action,action_out],axis=0)[1:]
            reward = np.concatenate([reward,[reward_out]],axis=0)[1:]
        elif model_type in ["CNN", "CNN_framestacking"]:
            state = state_out
            action = action_out
            reward = reward_out
        elif model_type == "CNN_LSTM":
            state = state_out
            action = action_out
            reward = reward_out
        
        img = cv2.cvtColor(np.uint8(env.render_obs()), cv2.COLOR_BGR2RGB)
        out.write(img)
        eval_res = log_on_step(
            eval_res,
            action[-1] if isinstance(action, np.ndarray) else action,
            reward[-1] if isinstance(reward, np.ndarray) else reward,
            env,
            info)
    
    env.close()
    out.release()
        
def main(test_config, best_models):
    #evaluate all models
    timestamp = str(datetime.now())

    for model_type, chkpt_path in tqdm(best_models.items(), total=len(best_models.keys())):

        # change the model
        mock_test_config = test_config
        mock_test_config["model_path"] = chkpt_path
        model_path, test_config_model, ray_config = model_initalizer(mock_test_config)
        
        # Set up env
        ray.init(**test_config_model.ray_init_config)
        register_env('Duckietown', launch_and_wrap_env)
        ModelCatalog.register_custom_model("custom_wrapped_cnn", Custom_wrapped_cnn)
        test_config_model.env_config["robot_speed"] = DEFAULT_ROBOT_SPEED
        temp_env_config = {
                "env": test_config_model.env_config,
                "wrapper": test_config_model.wrapper_config,
                "black_out": black_out_config
                }

        ppo_config = {
            "env": "Duckietown",  #define env
            "env_config": temp_env_config,
            }

        ppo_config.update(ray_config)
        trainer = ppo.PPOTrainer(
            config=ppo_config)
        trainer.restore(model_path)
        
        print(f"Loaded model from: {model_path}")
        print(f"with config:\n {trainer.config}")


        if test_config.calculate_nr_of_model_parameters:
            print(f"Model parameter calculation is in progess for {model_type}")
            show_model_statistics(ppo_config)

        if test_config.evaluate_custom_loop:
            env, obs, done = initalize_env(test_config_model)
            run_evaluation(model_type, trainer, env, obs, done, test_config_model, ray_config)
        
        if test_config.evaluate_official_loop:
            wandb_logger = init_logging(test_config_model, model_type, map=temp_env_config["env"]["map_name"])
            evaluator = DuckietownWorldEvaluator(
                temp_env_config,
                eval_lenght_sec=20,
                eval_map=temp_env_config["env"]["map_name"],
                model_type=model_type,
                ray_config=ray_config,
                wandb_logger=wandb_logger)
            evaluator.evaluate(trainer, f"evaluation_results/{timestamp}/{model_type}")
        ray.shutdown()


if __name__== "__main__":
    main(test_config, best_models)
