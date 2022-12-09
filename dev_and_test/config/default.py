'''
Config file to setup ray, ray_init, environment, RL
and model specific hyperparameters parameters for training.
'''

from datetime import datetime
timestamp = str(datetime.now())

# model options:
#   "CNN"   -> customCNN
#   "LSTM"  -> customCNN + LSTM
#   "TR"    -> customCNN + GTrXL

model_choice = "TR"

#######################################- OPTION 1: customCNN -#######################################

##ray configs
ray_config_cnn = {
    "framework": "torch",
    #PPO specific
    "gamma": 0.99,
    "num_workers": 4,
    "num_gpus": 0,
    "seed": 1234,
    "train_batch_size" : 4096,
    "lr" : 0.0001,
    "lr_schedule": [
            [0, 0.0001],
            [0.5e+6, 0.000000001],
        ],
    "lambda" : 0.95,
    "sgd_minibatch_size" : 128,
    "entropy_coeff": 0.0,
    "clip_param": 0.2,
    "grad_clip" : 0.5,
    "num_sgd_iter": 16,
    #"vf_clip_param" : 0.2,
    "evaluation_interval" : 10,
    "evaluation_num_episodes": 2,
    "evaluation_config": {
        "record_env": True
        }, #true
    #"_disable_execution_plan_api" : True,
    }

#checkpoint configs
chkpt_config_cnn = {
    "enabled" : False,
    "ckpth_path" : "/root/workdir/thesis/02_transformer/duckietown-RL-transformer-thesis/logs/LAST_SEMESTER_20220913-214924/PPOTrainer_Duckietown_ee9ba_00000_0_2022-09-13_21-49-24/checkpoint_000191/checkpoint-191",
    "overwrite_items" : {
        "ray_config": {"lr" : 0.00001, "num_workers":5},
        "wrapper_config" : {},
        "ray_stop_config" : {"timesteps_total" : 1.6e+6},
        "env_config" : {},
        "wandb_config" : {"name" : f"fine_tuning_{timestamp}"},
    }
}


####################################- OPTION 2: customCNN+LSTM -#####################################

##ray configs
ray_config_lstm = {
    "framework": "torch",
    "model": {
        "custom_model" : "custom_wrapped_cnn", #model config
        # Whether to wrap the model with an LSTM.
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 64,
        # Size of the LSTM cell.
        "lstm_cell_size": 954,
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": True,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": True,
        # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        "_time_major": False,
         "custom_model_config": {},
        },
    
    #PPO specific
    "gamma": 0.85,
    "num_workers": 4,
    "num_gpus": 0,
    "seed": 1234,
    "train_batch_size" : 4096,
    "lr" : 0.0005, #1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8,1 .e-9
                  #  x  |   x  |   x  |   x  |   x  |   x  |   x  |   x  |   x  |
    "lambda" : 0.8,
    "sgd_minibatch_size" : 512, #32 is optimal for PPO, but this cant be lower then 'max_seq_len'
    "entropy_coeff": 0.0,
    "clip_param": 0.1,
    "grad_clip" : 1.0,
    "num_sgd_iter": 16,
    #"vf_clip_param" : 0.2,
    "evaluation_interval" : 10,
    "evaluation_num_episodes": 2,
    "evaluation_config": {
        "record_env": True
        }, #true
    #"_disable_execution_plan_api" : True,
    }

#checkpoint configs
chkpt_config_lstm = {
    "enabled" : False,
    "ckpth_path" : "/root/workdir/thesis/02_transformer/duckietown-RL-transformer-thesis/logs/LAST_SEMESTER_20220913-214924/PPOTrainer_Duckietown_ee9ba_00000_0_2022-09-13_21-49-24/checkpoint_000191/checkpoint-191",
    "overwrite_items" : {
        "ray_config": {"lr" : 0.00001, "num_workers":5},
        "wrapper_config" : {},
        "ray_stop_config" : {"timesteps_total" : 1.6e+6},
        "env_config" : {},
        "wandb_config" : {"name" : f"fine_tuning_{timestamp}"},
    }
}


#################################- OPTION 3: customCNN+TRANSFORMER -#################################

##ray configs
ray_config_tr = {
    "framework": "torch",
    "model": {
        "custom_model" : "custom_wrapped_cnn", #model config
        "use_attention" : True,
        "max_seq_len": 64,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 8,
        # The input and output size of each transformer unit.
        "attention_dim": 128,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 3,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 128,
        # The memory sizes for inference and training.
        "attention_memory_inference": 64,
        "attention_memory_training": 64,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 512,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 3,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 3,
         "custom_model_config": {},
        },
    
    #PPO specific
    "gamma": 0.85,
    "num_workers": 4,
    "num_gpus": 0,
    "seed": 1234,
    "train_batch_size" : 4096,
    "lr" : 0.0001, #1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8,1 .e-9
                  #  x  |   x  |   x  |   x  |   x  |   x  |   x  |   x  |   x  |
    "lr_schedule": [
        [0, 0.0001],
        [1e+6, 0.00005],
        ],
    "lambda" : 0.8,
    "sgd_minibatch_size" : 512, #32 is optimal for PPO, but this cant be lower then 'max_seq_len'
    "entropy_coeff": 0.0,
    "clip_param": 0.1,
    "grad_clip" : 1,
    "num_sgd_iter": 16,
    #"vf_clip_param" : 0.2,
    "evaluation_interval" : 10,
    "evaluation_num_episodes": 2,
    "evaluation_config": {
        "record_env": True
        }, #true
    #"_disable_execution_plan_api" : True,
    }

#checkpoint configs
chkpt_config_tr = {
    "enabled" : False,
    "ckpth_path" : "logs/LAST_SEMESTER_20221019-051018/PPOTrainer_Duckietown_52bf8_00000_0_2022-10-19_05-10-18/checkpoint_000489/checkpoint-489",
    "overwrite_items" : {
        "ray_config_tr": {"num_workers":4, "num_gpus":0},
        "wrapper_config" : {},
        "ray_stop_config" : {"timesteps_total" : 2.5e+6},
        "env_config" : {},
        "wandb_config" : {"name" : f"fine_tuning_TR_blackout_{timestamp}"},
    }
}



###############################- GENERAL CONFIGS, USED FOR ALL MODELS -##############################
# wrapper configurations
wrapper_config = {
    "frame_stacking" : False,
    "frame_stacking_depth" : 0,
}

# blackout configurations
# if enabled -> hides observations at 'every_n_th' frame for 'length' frames long
black_out_config = {
    "enabled": False, # turn on/off
    "every_n_th": 100, # from every n-th frame
    "length": 30, # for length (default: 1)
    "visualize": False # to save video
}


##ray init config
ray_init_config = {
    "local_mode" : False, #easier debugging
    "object_store_memory" : 8737418240, # 8.GB
    "_memory": 2097152000,
    "_redis_max_memory": 209715200,
    "num_gpus": 0,
    }


##ray stop config
ray_stop_config = {
    "timesteps_total" : 2e+6,
}

##environment configs
env_config = {
    "seed" : 1234,
    "map_name" : "small_loop_bordered", #choose map
    "max_steps" : 500, #episode max steps
    "draw_curve" : False, #draw the lane following curve
    "draw_bbox" : False, #raw collision detection bounding boxes
    "domain_rand" : False, #enable domain randomization
    "frame_skip" : 1, # !!DO NOT CHANGE to 0, rather will give .nan rewards!! number of frames to skip
    "frame_rate" : 30,
    "randomize_maps_on_reset" : False, #default=False
    "distortion" : False,
    "camera_rand" : False,
    "dynamics_rand" : False, # enable dynamics randomization
    "accept_start_angle_deg" : 4, #4
    "full_transparency" : True,
}

##logging config
wandb_config = {
    "project" : "duckietown_transformer_thesis",
    "group" : "train",
    "entity" : "thesis-duckietown-transformer",
    "name" : f"train_{timestamp}",
    "reinit" : True,
    "dir" : "./logs",
    "save_code" : True,
    # "monitor_gym" : True, make video form environment
    }


