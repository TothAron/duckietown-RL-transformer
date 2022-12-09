"""
Reward wrappers

These modules are the modified versions of:
https://github.com/kaland313/Duckietown-RL
MIT License
Copyright (c) 2019 András Kalapos

- added WheelDiff penalty as option
- RewardVelocity tried with mean -> not promising result
"""

import gym
from gym_duckietown.simulator import NotInLane
import numpy as np


class DtRewardVelocity(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardVelocity, self).__init__(env)
        self.velocity_reward = 0.

    def reward(self, reward):
        self.velocity_reward = np.max(self.unwrapped.wheelVels) * 0.25
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            ###logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.velocity_reward

    def reset(self, **kwargs):
        self.velocity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['velocity'] = self.velocity_reward
        return observation, self.reward(reward), done, info

class DtRewardWheelDiff(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardWheelDiff, self).__init__(env)
        self.wheeldiff_reward = 0.

    def reward(self, reward):
        diff = np.clip(np.abs(np.diff(self.unwrapped.wheelVels))[0], 0., 1.)
        self.wheeldiff_reward = (1-diff)*0.1
        if np.isnan(self.wheeldiff_reward):
            self.wheeldiff_reward = 0.
            ###logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.wheeldiff_reward

    def reset(self, **kwargs):
        self.wheeldiff_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['wheeldiff_reward'] = self.wheeldiff_reward
        return observation, self.reward(reward), done, info


class DtRewardPosAngle(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardPosAngle, self).__init__(env)
            # gym_duckietown.simulator.Simulator

        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 1./2.
        self.orientation_reward = 0.

    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle
        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return -10.

        # print("Dist: {:3.2f} | Angle_deg: {:3.2f}".format(normed_lp_dist, normed_lp_angle))
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        ###logger.debug("Angle Narrow: {:4.3f} | Angle Wide: {:4.3f} ".format(angle_narrow_reward, angle_wide_reward))
        self.orientation_reward = self.scale * (angle_narrow_reward + angle_wide_reward)

        early_termination_penalty = 0.
        # If the robot leaves the track or collides with an other object it receives a penalty
        # if reward <= -1000.:  # Gym Duckietown gives -1000 for this
        #     early_termination_penalty = -10.
        return self.orientation_reward + early_termination_penalty

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    @staticmethod
    def gaussian(x, mu=0., sig=1.):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide

class DtRewardCollisionAvoidance(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardCollisionAvoidance, self).__init__(env)
            # gym_duckietown.simulator.Simulator
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.

    def reward(self, reward):
        # Proximity reward is proportional to the change of proximity penalty. Range is ~ 0 - +1.5 (empirical)
        # Moving away from an obstacle is promoted, if the robot and the obstacle are close to each other.
        proximity_penalty = self.unwrapped.proximity_penalty2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        self.proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * 50
        if self.proximity_reward < 0.:
            self.proximity_reward = 0.
        #logger.debug("Proximity reward: {:.3f}".format(self.proximity_reward))
        self.prev_proximity_penalty = proximity_penalty
        return reward + self.proximity_reward

    def reset(self, **kwargs):
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['collision_avoidance'] = self.proximity_reward
        return observation, self.reward(reward), done, info
