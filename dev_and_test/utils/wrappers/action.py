"""
Action wrappers

These modules are the modified versions of:
https://github.com/kaland313/Duckietown-RL
MIT License
Copyright (c) 2019 András Kalapos
"""

import gym
import numpy as np
from gym import spaces

class Heading2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env, heading_type=None):
        super(Heading2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        if self.heading_type == 'heading_trapz':
            straight_plateau_half_width = 0.3333  # equal interval for left, right turning and straight
            self.mul = 1. / (1. - straight_plateau_half_width)

    def action(self, action):

        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):


            #TODO: itt lehet a probléma, az action a hálóból vektorként jön ki 
            # a sequence modelleknél és itt rossz van kivéve
            #action = action[0]
            action = action[-1]
        # action = [-0.5 * action + 0.5, 0.5 * action + 0.5]
        if self.heading_type == 'heading_smooth':
            action = np.clip(np.array([1 + action ** 3, 1 - action ** 3]), 0., 1.)  # Full speed single value control
        elif self.heading_type == 'heading_trapz':
            action = np.clip(np.array([1 - action, 1 + action]) * self.mul, 0., 1.)
        elif self.heading_type == 'heading_sine':
            action = np.clip([1 - np.sin(action * np.pi), 1 + np.sin(action * np.pi)], 0., 1.)
        elif self.heading_type == 'heading_limited':
            action = np.clip(np.array([1 + action*0.666666, 1 - action*0.666666]), 0., 1.)
        else:
            action = np.clip(np.array([1 + action, 1 - action]), 0., 1.)
        return action