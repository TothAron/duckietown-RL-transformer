"""
Observation wrappers

These modules are the modified versions of:
https://github.com/kaland313/Duckietown-RL
MIT License
Copyright (c) 2019 András Kalapos

- added blackout wrapper as option
- fixed resize wrapper
"""

import gym
import cv2
from gym import spaces
import numpy as np


class ClipImageWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(ClipImageWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        # Region Of Interest
        # r = [margin_left, margin_top, width, height]
        self.roi = [0, top_margin, img_width, img_height]

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        r = self.roi
        observation = observation[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        return observation


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=84, resize_h=84):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [ resize_h, resize_w, obs_shape[2]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation

    def reset(self):
        obs = gym.ObservationWrapper.reset(self)
        obs = cv2.resize(obs, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)

        return obs 

    def step(self, actions):
        obs, reward, done, info = gym.ObservationWrapper.step(self, actions)
        env = (
            cv2.resize(obs,
                    dsize=(self.resize_w, self.resize_h),
                    interpolation=cv2.INTER_CUBIC),
            reward,
            done,
            info,
        )

        return env


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class BlackOutWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, every_n_th=30, length=1, visualize=False):
        super(BlackOutWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

        # black out specific
        self.start_counter: int = length - every_n_th #skips immediately blackout at start
        self.every_n_th = every_n_th
        self.length = length

        # video save specific
        self.visualize = visualize
        if self.visualize:
            cap = cv2.VideoCapture(0)
            self.video_writer = cv2.VideoWriter(
                f'black_out_test_every{self.every_n_th}_length{self.length}.mp4',
                cv2.VideoWriter_fourcc(*'MP4V'),
                30.0,
                (img_width, img_height)
            )

    def observation(self, obs):
        if ((self.start_counter % self.every_n_th) < self.length) and (self.start_counter >= 0):
                #black out frame
                obs *= 0
        self.start_counter += 1

        if self.visualize:
            self._visualize(obs)
        return obs


    def reset(self,**kwargs):
        observation = self.env.reset(**kwargs) 
        img_height, img_width, depth = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)
        self.start_counter: int = self.length - self.every_n_th
        if self.visualize:
            self.video_writer.release()
            self.video_writer = cv2.VideoWriter(
                f'black_out_test_every{self.every_n_th}_length{self.length}.mp4',
                cv2.VideoWriter_fourcc(*'MP4V'),
                30.0,
                (img_width, img_height)
            )
        return self.observation(observation)

    
    def _visualize(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        self.video_writer.write(obs)
        


class ObservationBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=3):
        super(ObservationBufferWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)

        # The last dimension, is used. For images, this should be the depth.
        # For vectors, the output is still a vector, just concatenated.
        self.buffer_axis = len(obs_space_shape_list) - 1
        obs_space_shape_list[self.buffer_axis] *= obs_buffer_depth
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [obs_space_shape_list[0], obs_space_shape_list[1], obs_space_shape_list[2]],
            dtype=self.observation_space.dtype,
        )

        if len(self.observation_space.shape) == 3:
            limit_low = self.observation_space.low[0, 0, 0]
            limit_high = self.observation_space.high[0, 0, 0]
        elif len(self.observation_space.shape) == 1:
            # Note this was implemented for vector like observation spaces (e.g. a VAE latent vector)
            limit_low = self.observation_space.low[0]
            limit_high = self.observation_space.high[0]
        else:
            assert False, "Only 1 or 3 dimentsional obs space supported!"

        self.observation_space = spaces.Box(
            limit_low,
            limit_high,
            tuple(obs_space_shape_list),
            dtype=self.observation_space.dtype)
        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        if self.obs_buffer_depth == 1:
            return obs
        if self.obs_buffer is None:
            self.obs_buffer = np.concatenate([obs for _ in range(self.obs_buffer_depth)], axis=self.buffer_axis)
        else:
            self.obs_buffer = np.concatenate((self.obs_buffer[..., (obs.shape[self.buffer_axis]):], obs), axis=self.buffer_axis)
        return self.obs_buffer

    def reset(self, **kwargs):
        self.obs_buffer = None
        observation = self.env.reset(**kwargs)
        return self.observation(observation)
        
