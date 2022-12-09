"""
DuckietownWorldEvaluator
"""

from cmath import pi
import os
import numpy as np
import json
import time
import logging
import copy
from ray.tune.logger import pretty_print
import cv2

from duckietown_world import SE2Transform
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric, make_timeseries
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.world_duckietown.map_loading import load_map

from gym_duckietown.simulator import DEFAULT_ROBOT_SPEED
from utils.env import launch_and_wrap_env
from utils.trajectory_plot import correct_gym_duckietown_coordinates

DEFAULT_EVALUATION_MAP = 'ETHZ_autolab_technical_track'

logger = logging.getLogger(__name__)


class DuckietownWorldEvaluator:
    """
    Evaluates a RLlib agent using the same evaluator which is used in DTS evaluate.
    To adapt for a different agent implementation than RLlib, __init__ and _compute_action,
    (_collect_trajectory) should be modified.
    """
    # These start poses are exactly the same as used by dts evaluate
    tile_size=0.585
    # Pose description [[x, 0, z], rot].
    # x, z are measured in meters (x: horizontal, z: vertical, [0,0,0]: top left corner), rot is measured in radians
    start_poses = {'ETHZ_autolab_technical_track': [[[0.7019999027252197, 0, 0.41029359288296474], -0.2687807048071267],
                                                    [[0.44714101540138385, 0, 2.2230001401901243], 1.31423292675173],
                                                    [[1.5552862449923595, 0, 1.0529999446868894], 1.4503686084072878],
                                                    [[1.6380000114440918, 0, 3.0929162652880935], -3.0264009229581674],
                                                    [[0.3978251477698904, 0, 2.8080000591278074], 1.2426744274199626],
                                                    # [[0.2, 0, 2.8], np.pi/2*1.1],  # For testing lane-correction
                                                    # [[0.585, 0, 5.75*0.585], np.pi],  # For testing lane-correction
                                                    ],
                   'my_map_curvy': [
                    [[1.75*tile_size, 0, 9.5*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 10.5*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 10.0*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 10.2*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 10.1*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 9.6*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 9.7*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 9.8*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 9.9*tile_size], np.pi / 2],
                    #[[1.75*tile_size, 0, 9.4*tile_size], np.pi / 2],
                   ],
                   'my_map_straight': [
                    [[3.75*tile_size, 0, 8.5*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 8.6*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 8.7*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 7.5*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 7.6*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 7.7*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 6.5*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 6.6*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 6.7*tile_size], np.pi / 2],
                    #[[3.75*tile_size, 0, 6.8*tile_size], np.pi / 2],
                   ]
    
                   }

    def __init__(self, temp_env_config, eval_lenght_sec=15, eval_map=DEFAULT_EVALUATION_MAP, model_type=None, ray_config=None, wandb_logger=None):
        self.wandb_logger = wandb_logger
        _temp_env_config = copy.deepcopy(temp_env_config)
        _env_config =  _temp_env_config["env"]
        # An official evaluation episode is 15 seconds long
        _env_config['max_steps'] = eval_lenght_sec * _env_config['frame_rate']+1
        _env_config['robot_speed'] = DEFAULT_ROBOT_SPEED
        # Agets should be evaluated on the official eval map
        _env_config['map_name'] = eval_map
        self.map_name = _env_config['map_name']
        # Make testing env
        _temp_env_config["env"]= _env_config
        self.env = launch_and_wrap_env(_temp_env_config, one_map=self.map_name)

        self.model_type = model_type
        self.ray_config = ray_config 
        # Set up evaluator
        # Creates an object 'duckiebot'
        self.ego_name = 'duckiebot'
        self.db = DB18()  # class that gives the appearance
        # load one of the maps
        self.dw = load_map(self.map_name)

    def evaluate(self, agent, outdir, episodes=None):
        """
        Evaluates the agent on the map inicialised in __init__
        :param agent: Agent to be evaluated, passed to self._collect_trajectory(agent,...)
        :param outdir: Directory for logged outputs (trajectory plots + numeric data)
        :param episodes: Number of evaluation episodes, if None, it is determined based on self.start_poses
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if episodes is None:
            episodes = len(self.start_poses.get(self.map_name, []))
        totals = {}
        added_logs = {
            "mean_inference_time": [],
            "mean_env_time": [],
            "reward_return": [],
            "mean_robot_speed": []
        }

        for i in range(episodes):
            episode_path, episode_orientations, \
                episode_timestamps, episode_inference_times, \
                episode_env_step_times, episode_rewards, episode_robot_speeds = \
                self._collect_trajectory(agent, i, outdir)

            added_logs["mean_inference_time"].append(np.mean(episode_inference_times))
            added_logs["mean_env_time"].append(np.mean(episode_env_step_times))
            added_logs["reward_return"].append(np.sum(episode_rewards))
            added_logs["mean_robot_speed"].append(np.mean(episode_robot_speeds))

            logger.info("Episode {}/{} sampling completed".format(i+1, episodes))
            if len(episode_timestamps) <= 1:
                continue
            episode_path = np.stack(episode_path)
            episode_orientations = np.stack(episode_orientations)
            # Convert them to SampledSequences
            transforms_sequence = []
            for j in range(len(episode_path)):
                transforms_sequence.append(SE2Transform(episode_path[j], episode_orientations[j]))
            transforms_sequence = SampledSequence.from_iterator(enumerate(transforms_sequence))
            transforms_sequence.timestamps = episode_timestamps

            _outdir = outdir
            if outdir is not None:
                _outdir = os.path.join(outdir, "Trajectory_{}".format(i+1))
            evaluated, to_be_logged_lists = self._eval_poses_sequence(transforms_sequence, outdir=_outdir)
            logger.info("Episode {}/{} plotting completed".format(i+1, episodes))
            totals = self._extract_total_episode_eval_metrics(evaluated, totals, display_outputs=True)

        # Calculate the median total metrics
        median_totals = {}
        mean_totals = {}
        stdev_totals = {}
        for key, value in totals.items():
            median_totals[key] = np.median(value)
            mean_totals[key] = np.mean(value)
            stdev_totals[key] = np.std(value)

        # for json logging
        mean_inference_time = np.mean(added_logs["mean_inference_time"])
        std2_inference_time = 2 * np.std(added_logs["mean_inference_time"])
        mean_env_step_time =  np.mean(added_logs["mean_env_time"])
        std2_env_step_time = 2 * np.std(added_logs["mean_env_time"])
        mean_reward_return = np.mean(added_logs["reward_return"])
        std2_reward_return = 2 * np.std(added_logs["reward_return"])
        mean_robot_speed = np.mean(added_logs["mean_robot_speed"])
        std2_robot_speed = 2 * np.std(added_logs["mean_robot_speed"])
        
        # only for wandb logging
        to_be_logged_lists["reward"] = episode_rewards
        to_be_logged_lists["robot_speed"] = episode_robot_speeds
        to_be_logged_lists["inference_time"] = episode_inference_times
        to_be_logged_lists["env_step_time"] = episode_env_step_times

        for i in range(len(to_be_logged_lists["reward"])-1):
            temp_log = {}
            for key, value in to_be_logged_lists.items():
                temp_log[key] = value[i]
            self.wandb_logger.log(temp_log, step=i+1)

        # Save results to file
        with open(os.path.join(outdir, "total_metrics.json"), "w") as json_file:
            json.dump({'median_totals': median_totals,
                       'mean_totals': mean_totals,
                       'stdev_totals': stdev_totals,
                       'episode_totals': totals,
                       'mean_inference_time': mean_inference_time,
                       'std2_inference_time': std2_inference_time,
                       'mean_env_step_time': mean_env_step_time,
                       'std2_env_step_time': std2_env_step_time,
                       'mean_reward_return': mean_reward_return,
                       'std2_reward_return': std2_reward_return,
                       'mean_robot_speed': mean_robot_speed, 
                       'std2_robot_speed': std2_robot_speed
                       }, json_file, indent=2)

        logger.info("\nMedian total metrics: \n {}".format(pretty_print(median_totals)))
        logger.info("\nMean total metrics: \n {}".format(pretty_print(mean_totals)))
        logger.info("\nStandard deviation of total metrics: \n {}".format(pretty_print(stdev_totals)))

    def _collect_trajectory(self, agent, i, outdir):
        episode_path = []
        episode_orientations = []
        episode_timestamps = []
        episode_inference_times = []
        episode_env_step_times = []
        episode_rewards = []
        episode_robot_speeds = []

        if self.map_name in self.start_poses.keys() and i < len(self.start_poses.get(self.map_name, [])):
            self.env.unwrapped.user_tile_start = [0, 0]
            self.env.unwrapped.start_pose = self.start_poses[self.map_name][i]
        else:
            # No (more) preselected start positions are available -> gym_duckietown should generate them randomly
            # For that user_tile_start and start_pose must be None
            self.env.unwrapped.user_tile_start = None
            self.env.unwrapped.start_pose = None
        obs = self.env.reset()
        done = False
        out = cv2.VideoWriter(f"{outdir}/{self.model_type}.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))

        #init_RL_params
        if self.model_type == "CNN_TR":
            num_transformers = self.ray_config["model"]["attention_num_transformer_units"]
            attention_dim = self.ray_config["model"]["attention_dim"]
            memory = self.ray_config["model"]["attention_memory_inference"]
            state = [np.zeros([memory, attention_dim], np.float32) for _ in range(num_transformers)]
            action = np.zeros(self.ray_config["model"]["attention_use_n_prev_actions"],)
            reward = np.zeros(self.ray_config["model"]["attention_use_n_prev_rewards"],)
            seq_len = num_transformers
        elif self.model_type in ["CNN", "CNN_framestacking"]:
            state = np.zeros(1)
            action = np.zeros(1)
            reward = np.zeros(1)
            seq_len = 0
        elif self.model_type == "CNN_LSTM":
            seq_len = self.ray_config["model"]["max_seq_len"]
            cell_size = self.ray_config["model"]["lstm_cell_size"]
            state = [np.zeros([cell_size], np.float32) for _ in range(2)]
            action = np.zeros(1)
            reward = np.zeros(1)
        while not done:
            t_0 = time.time()
            action_out, state_out, _ = agent.compute_single_action(
            observation=obs,
            prev_action=action,
            prev_reward = reward,
            state=state,
            full_fetch=True,
            explore=False,
            policy_id="default_policy")
            t_1 = time.time()
            obs, reward_out, done, info = self.env.step(action_out)
            t_2 = time.time()
            cur_pos = correct_gym_duckietown_coordinates(self.env.unwrapped, self.env.unwrapped.cur_pos)
            episode_path.append(cur_pos)
            episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
            episode_timestamps.append(info['Simulator']['timestamp'])
            episode_inference_times.append((t_1-t_0)*1000)
            episode_env_step_times.append((t_2-t_1)*1000)
            episode_rewards.append(reward_out[-1] if isinstance(reward_out, np.ndarray) else reward_out)
            episode_robot_speeds.append(info["Simulator"]["robot_speed"])
            
            # concatenate rl variables if needed
            if self.model_type in "CNN_TR":
                state = [np.concatenate([state[i], [state_out[i]]], axis=0)[1:] for i in range(seq_len)]
                action = np.concatenate([action,action_out],axis=0)[1:]
                reward = np.concatenate([reward,[reward_out]],axis=0)[1:]
            elif self.model_type in ["CNN", "CNN_framestacking", "CNN_LSTM"]:
                state = state_out
                action = action_out
                reward = reward_out

            #info dict--->

            #{'action': [array([0.73350906]), array([1.])], 
            #'lane_position': {
            #    'dist': 0.003171537702279681, 
            #    'dot_dir': 0.9652653626271975,
            #    'angle_deg': 15.145519901211781,
            #    'angle_rad': 0.2643391892024719}, 
            #'robot_speed': 0.1700086467788305,
            #'proximity_penalty': 0,
            #'cur_pos': [
            #    0.7106803677718864,
            #    0.0,
            #    0.4126715377022796],
            #'cur_angle': -0.2643391892024718,
            #'wheel_velocities': [
            #    array([0.88021088]),
            #    array([1.2])],
            #'timestamp': 0.19999999999999998,
            #'tile_coords': [1, 0],
            #'msg': ''}
            img = cv2.cvtColor(np.uint8(self.env.render_obs()), cv2.COLOR_BGR2RGB)
            out.write(img)
            print(info["Simulator"]["robot_speed"])
        self.env.unwrapped.start_pose = None
        self.user_tile_start = None
        out.release()
        return episode_path, episode_orientations, \
            episode_timestamps, episode_inference_times, \
            episode_env_step_times, episode_rewards, episode_robot_speeds


    def _eval_poses_sequence(self, poses_sequence, outdir=None):
        """
        :param poses_sequence:
        :param outdir: If None evaluation outputs plots won't be saved
        :return:
        """
        # puts the object in the world with a certain "ground_truth" constraint
        self.dw.set_object(self.ego_name, self.db, ground_truth=poses_sequence)
        # Rule evaluation (do not touch)
        interval = SampledSequence.from_iterator(enumerate(poses_sequence.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence,
                                   interval=interval, world=self.dw, ego_name=self.ego_name)
        if outdir is not None:
            timeseries = make_timeseries(evaluated)
            to_be_logged_lists = {
                "deviation-heading": np.array(timeseries['rules/deviation-heading'].sequences["incremental"].values)*180/pi,
                "in-drivable-lane": np.array(timeseries['rules/in-drivable-lane'].sequences["incremental"].values),
                "deviation-center-line": np.array(timeseries['rules/deviation-center-line'].sequences["incremental"].values),
                "driving-distance-any": np.array(timeseries['rules/driving-distance/driven_any'].sequences["incremental"].values),
                "driving-distance-lanedir": np.array(timeseries['rules/driving-distance/driven_lanedir'].sequences["incremental"].values),
                "driven_lanedir_consec": np.array(timeseries['rules/driving-distance-consecutive/driven_lanedir_consec'].sequences["incremental"].values),
                "distance-from-start": np.array(timeseries['rules/distance-from-start'].sequences["incremental"].values),
            }
            draw_static(self.dw, outdir, timeseries=timeseries)
        print(self.dw.get_drawing_children())
        self.dw.remove_object(self.ego_name)
        #self.dw.remove_object('tilemap')
        return evaluated, to_be_logged_lists

    @staticmethod
    def _extract_total_episode_eval_metrics(evaluated, totals, display_outputs=False):
        episode_totals = {}
        for k, rer in evaluated.items():
            from duckietown_world.rules import RuleEvaluationResult
            assert isinstance(rer, RuleEvaluationResult)
            for km, evaluated_metric in rer.metrics.items():
                assert isinstance(evaluated_metric, EvaluatedMetric)
                episode_totals[k] = evaluated_metric.total
                if not (k in totals):
                    totals[k] = [evaluated_metric.total]
                else:
                    totals[k].append(evaluated_metric.total)
        if display_outputs:
            logger.info("\nEpisode total metrics: \n {}".format(pretty_print(episode_totals)))

        return totals