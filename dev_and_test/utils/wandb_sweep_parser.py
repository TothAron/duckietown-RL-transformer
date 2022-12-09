"""

The script to update the config dict and the config file for ongoing training 
with parameters what wandb sweep pass (right at the beggining).

"""


# Necessary for wandb sweep
import argparse

def wandb_sweep_hps_override(config):
    parser = argparse.ArgumentParser() 

    #add parameters you want to sweep, here and in wandb sweep online config:
    parser.add_argument("--sweep", type=bool, default=False) #it is false by default
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coeff", type=float, default= 0.0)
    parser.add_argument("--clip_param", type=float, default=0.2)

    args = parser.parse_args()

    
    if args.sweep:

        # following is needed to sort the '--something.something value' 
        # arguments into nested dict
        new_params={}
        plain_wandb_config={}
        args_dict = vars(args)
        for each in args_dict.keys():
            splitted = each.split(".")

            if splitted[0] == "sweep": #dont push the 'is sweep is on' param to ray rllib
                continue

            if len(splitted)==2:
                #new_params -> only used for printing new hps
                #plain_wandb_config -> only used for overwrite saved config file in utils/saver.py
                if splitted[0] not in new_params:
                    new_params.update( {splitted[0]: {splitted[1] :args_dict[each]}})
                    plain_wandb_config.update({splitted[1] :args_dict[each]})

                else:
                    new_params[splitted[0]].update({splitted[1] :args_dict[each]})
                    plain_wandb_config.update({splitted[1] :args_dict[each]})

                #update original config
                config[splitted[0]].update({splitted[1] :args_dict[each]})
            else:
                new_params.update( {splitted[0]: args_dict[each]})
                plain_wandb_config.update({splitted[0] :args_dict[each]})
                config.update({splitted[0]: args_dict[each]})

        #update the saved default config file in the current log folder
        

        print("new hyperparameters: ", new_params)
        #print("full_config: ",config)
        print("WANDB SWEEP PASSED HYPERPARAMETERS!")
        return config, plain_wandb_config
    else:
        print("WANDB SWEEP IS NOT PASSED HYPERPARAMETERS! If you train manually, ignore this message.")
        return config, None