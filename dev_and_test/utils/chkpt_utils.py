"""
Checkpointing tools for load and initalize fine-tuning from saved checkpoints.
"""
import importlib.util
import sys
import collections

def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def config_and_chkpt_initalizer(config):

    # select and load model specific config
    choice = config.model_choice.lower()
    config.wandb_config["name"] = f"{choice.upper()}_{config.wandb_config['name']}"

    ray_config = getattr(config,f"ray_config_{choice}")
    chkpt_config = getattr(config,f"chkpt_config_{choice}")
    


    if chkpt_config["enabled"]:
        chkpt_path = chkpt_config["ckpth_path"]
        config_path = f"{'/'.join(chkpt_path.split('/')[:-3])}/save/dev_and_test/config/default.py"

        # import saved config
        spec = importlib.util.spec_from_file_location("default", config_path)
        default = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(default)
        processed_config = default

        # overwrite arguments for finetune
        for key, val in chkpt_config["overwrite_items"].items():
            each_dict = getattr(processed_config, key)
            updated_dict = update_dict(each_dict, val)
            setattr(processed_config, key, updated_dict)
        

    else:
        # no changes
        chkpt_path = None
        processed_config = config
        
    return processed_config, ray_config, chkpt_config, chkpt_path 

def model_initalizer(config):
    model_path = config["model_path"]
    config_path = f"{'/'.join(model_path.split('/')[:-3])}/save/dev_and_test/config/default.py"

    # import saved config
    spec = importlib.util.spec_from_file_location("test_config", config_path)
    test_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_config)
    processed_config = test_config

    model_type = test_config.model_choice.lower()

    updates = {
        "env_config": {"map_name": config["map_name"], "domain_rand": False, "distortion": False},
        f"ray_config_{model_type}": {"num_workers": 1, "num_gpus": 0, "explore": False},
        "ray_init_config": {"local_mode": True, "num_gpus":0},
            }

    # overwrite arguments for test
    for key, val in updates.items():
        each_dict = getattr(processed_config, key)
        updated_dict = update_dict(each_dict, val)
        setattr(processed_config, key, updated_dict)

    ray_config = getattr(processed_config,f"ray_config_{model_type}") 
    return model_path, processed_config, ray_config