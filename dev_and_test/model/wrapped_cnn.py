import os
import torch
import numpy as np
import copy
from typing import Dict, List
import gym
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

# The custom model that will be wrapped by an LSTM.
class Custom_wrapped_cnn(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        #define output's length of the CNN model standard: [256,]
        #self._batch_size = 64 #same as sgd_mini_batch_size

        #self.num_outputs = self._batch_size
        self.num_outputs = 256
        self.last_layer_is_flattened = True

        #print(self.obs_space.shape)
        #print(self.num_outputs)
        self._last_batch_size = None
        self._logits = None

        self._base_layers = [       #IN: [B, 84,84,3]
            SlimConv2d(           
                    in_channels=3,
                    out_channels=16,
                    kernel=8,
                    stride=4,
                    padding=[2,2,2,2],
                    activation_fn=nn.ReLU,
                    ),             #conv_1_out: [B,21,21,16]
            SlimConv2d(            
                    in_channels=16,
                    out_channels=32,
                    kernel=4,
                    stride=2,
                    padding=[1,2,1,2],
                    activation_fn=nn.ReLU,
                    ),             #conv_2_out: [B,11,11,32]
            SlimConv2d(
                    in_channels=32,
                    out_channels=self.num_outputs,
                    kernel=11,
                    stride=1,
                    padding=None,
                    activation_fn=nn.ReLU,
                    )             #conv_3_out: [B,1,1,256]      
                ]
        

        self._action_head = nn.Flatten()  #flatten_out: [B,256] <--------Action model out (vision model)


        self._value_head = SlimConv2d(
                            in_channels=self.num_outputs,
                            out_channels=1,
                            kernel=1,
                            stride=1,
                            padding=None,
                            activation_fn=None
                            )       #value_head_out = [B,1,1,1]  <--------Value model out

        #build action model layers
        self._action_model = copy.deepcopy(self._base_layers)
        self._action_model.append(self._action_head)

        #build value model layers
        self._value_model = copy.deepcopy(self._base_layers)
        self._value_model.append(self._value_head)

        #build vison model and value model
        self._base_model_action = nn.Sequential(*self._action_model)
        self._base_model_value = nn.Sequential(*self._value_model)
    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):

        self._obs = input_dict["obs"].float()
        self._obs = self._obs.permute(0,3,1,2) #make channel last format
        # Store last batch size for value_function output.
        self._last_batch_size = self._obs.shape[0]
        #print(obs.shape)
        
        self._action_out = self._base_model_action(self._obs)
        

        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        
        return self._action_out, state

        # Return CNN value net output
    @override(TorchModelV2)
    def value_function(self) -> TensorType:

        #shared layers
        share_layers = False
        if share_layers:
            assert False, "No vf_share_layers implemented in custom model."
            # self._value_head_input = self._vision_out
            # self._logits = self._value_head(self._value_head_input)
            # self._logits = self._logits.squeeze(3) # makes [?, 1, 1, 1]-> [?,1,1]
            # self._logits = self._logits.squeeze(2) # -> [?,1]
            # self._logits = self._logits.squeeze(1) # -> [?]
        else:
            self._logits = self._base_model_value(self._obs)
            self._logits = self._logits.squeeze(3) # makes [?, 1, 1, 1]-> [?,1,1]
            self._logits = self._logits.squeeze(2) # -> [?,1]
            self._logits = self._logits.squeeze(1) # -> [?]

        return self._logits