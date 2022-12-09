"""
Wandb custom logger class for logging various metrics from ray RLlib.
"""

import wandb
import logging
import ray.tune.logger
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIMESTEPS_TOTAL)
from ray.tune.utils import flatten_dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.trajectory_plot import plot_trajectories

logger = logging.getLogger(__name__)


class WeightsAndBiasesLogger(ray.tune.logger.Logger):
    def __init__(self, config, logdir, trial):
        super(WeightsAndBiasesLogger, self).__init__(config, logdir, trial)
        # logger.warning("WeightsAndBiasesLogger.__init__() called! Trial.experiment_tag: {}".format(trial.experiment_tag))

        self.trial = trial
        self.experiment_tag = trial.experiment_tag

        valid_config = config.copy()
        wandb_config = valid_config['callbacks']["wandb_config_pass"]
        self.wandb_run = wandb.init(**wandb_config)
        del valid_config['callbacks']
        valid_config = flatten_dict(valid_config, delimiter="/")
        self.wandb_run.config.update(valid_config, allow_val_change=True)

    def on_result(self, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        # Log scalars
        logged_results = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean',
                          'custom_metrics', 'sampler_perf', 'info', 'perf']
        result_copy = result.copy()
        for key, val in result.items():
            if key not in logged_results:
                del result_copy[key]
        flat_result = flatten_dict(result_copy, delimiter="/")
        self.wandb_run.log(flat_result, step=step, sync=False)

        # Log histograms
        for key, val in result['hist_stats'].items():
            try:
                if key != '_robot_coordinates':
                    self.wandb_run.log({"Histograms/"+key: wandb.Histogram(val)}, step=step)
            except ValueError:
                logger.warning("Unable to log histogram for {}".format(key))

        # Log trajectories
        traj_fig = plot_trajectories(result['hist_stats']['_robot_coordinates'])
        #traj_fig.savefig("Trajectory.png")
        self.wandb_run.log({'Episode Trajectories': wandb.Image(traj_fig)}, step=step)
        plt.close(traj_fig)

    def close(self):
        wandb.join()
