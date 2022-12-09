"""
Config file to adjust test mode parameters.
"""

# ~ SET.0 ~
# Set evaluation options.
calculate_nr_of_model_parameters = False
evaluate_custom_loop = False
evaluate_official_loop = True # <- use this for rich info and trajectory


# ~ SET.1 ~
# Choose the map you want to test on.
test_config = {
    "map_name": "my_map_straight"
}


# ~ SET.2 ~
# To use observation blackout during testing.
# If enabled=False, than every observation will be remain as the agent observs them.
# IF enabled=True, hides observations at 'every_n_th' frame for 'length' frames long.
black_out_config = {
    "enabled": False, # turn on/off
    "every_n_th": 100, # from every n-th frame
    "length": 30, # for length (default: 1)
    "visualize": False # to save video
}


# ~ SET.3 ~
# Add yout best model from each type here
# and the script will automatically evaluate these.
best_models = {
    "CNN": "logs/LAST_SEMESTER_20221021-153420/PPOTrainer_Duckietown_d546b_00000_0_2022-10-21_15-34-21/checkpoint_000489/checkpoint-489",
    "CNN_framestacking": "logs/LAST_SEMESTER_20221025-093453/PPOTrainer_Duckietown_477ee_00000_0_2022-10-25_09-34-53/checkpoint_000489/checkpoint-489",
    "CNN_LSTM": "logs/LAST_SEMESTER_20221020-194735/PPOTrainer_Duckietown_0b98e_00000_0_2022-10-20_19-47-35/checkpoint_000489/checkpoint-489",
    "CNN_TR": "logs/LAST_SEMESTER_20221019-051018/PPOTrainer_Duckietown_52bf8_00000_0_2022-10-19_05-10-18/checkpoint_000489/checkpoint-489"
}
