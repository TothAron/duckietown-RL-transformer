"""

Based on Ray RLlib's KERAS GTrXL implementation at 2022/04/19
https://github.com/ray-project/ray/blob/master/rllib/models/tf/attention_net.py

"""

import gym
from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Any, Dict, Optional, Type, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.layers import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)


from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.models.tf.attention_net import Keras_AttentionWrapper
tf1, tf, tfv = try_import_tf()

#custom CNN
from ray.rllib.models.tf.misc import normc_initializer


class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=8,
            strides=4,
            padding="same",
            name="conv_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(self.inputs)
        layer_2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=4,
            strides=2,
            padding="same",
            name="conv_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(layer_1)
        layer_3 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=11,
            strides=1,
            padding="same",
            name="conv_layer3",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(layer_2)
        layer_out = tf.keras.layers.Flatten(
            name="flatten"
        )(layer_3)

        
        # cnn_last = gym.spaces.Box(
        #     float("-inf"), float("inf"), shape=(layer_out.shape,), dtype=np.float32
        # )

        # layer_gtrxl = Keras_AttentionWrapper(
        #     input_space = layer_out,
        #     action_space = gym.spaces.Space,
        #     #*,
        #     name = "gtrxl",
        #     max_seq_len=20,
        #     num_transformer_units=1,
        #     attention_dim=64,
        #     num_heads=2,
        #     memory_inference=50,
        #     memory_training=50,
        #     head_dim=32,
        #     position_wise_mlp_dim=32,
        #     init_gru_gate_bias=2.0
        #     )

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_3)
        self.base_model = tf.keras.Model(self.inputs, layer_out)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    # def value_function(self):
    #     return tf.reshape(self._value_out, [-1])

    # def metrics(self):
    #     return {"foo": tf.constant(42.0)}


