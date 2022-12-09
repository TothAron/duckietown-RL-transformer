from config import default
import datetime
import os


def save_scripts(plain_wandb_config):
    global_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_name = default.wandb_config["group"] + "_" + str(global_tag)
    os.system('mkdir -p ./logs/{}/save/'.format(train_name))

    #os.system('cp -ar ./dev_and_test/config /logs/{}/save/config '.format(train_name))
    #os.system('cp -ar ./dev_and_test/model /logs/{}/save/model '.format(train_name))
    #os.system('cp -ar ./dev_and_test/utils /logs/{}/save/utils '.format(train_name))
    os.system('cp -ar ./dev_and_test ./logs/{}/save '.format(train_name))
    

    # update config file if wandb sweep used 
    # (in case wandb updated config dict -> saved config file will be updated (neccessary))
    if plain_wandb_config != None:
        config_file = open("logs/{}/save/dev_and_test/config/default.py".format(train_name), 'r')
        data = config_file.readlines()
        
        for idx, line in enumerate(data):
            for key in plain_wandb_config.keys():
                if "        \"{}\"".format(key) in line:
                    data[idx] = "\t\t\""+key+"\" : "+str(plain_wandb_config[key])+", # !UPDATED BY WANDB SWEEP!\n"
                elif "    \"{}\"".format(key) in line:
                    data[idx] = "\t\""+key+"\" : "+str(plain_wandb_config[key])+", # !UPDATED BY WANDB SWEEP!\n"
                
        with open("logs/{}/save/dev_and_test/config/default.py".format(train_name), 'w') as file:
            file.writelines(data)

    print("Saved scripts to log folder.")

    return train_name