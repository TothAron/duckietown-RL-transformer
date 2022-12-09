# Duckietown RL transformer

 

<img src="https://cdn.shopify.com/s/files/1/0053/4439/5336/files/duckie-only-transparent_1200x1200.png?v=1553620419" align="right"
     alt="Duckietown logo" width="170" height="170">

The **duckietown-RL-transformer** repository provides solution for lane following in the Duckietown environment.
It includes train, evaluation and test script for a hybrid **CNN - Transformer** ([GTrXL](https://arxiv.org/abs/1910.06764)) 
machine learning model, which is supposed to control duckiebot.
The solution builds upon the [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) framework for reinforcement learning (RL)
and [Weights & Biases](https://wandb.ai) framework for logging.




|<img src=".readme_extras/demo_sim.gif" alt="Demonstration" width="380" height="400">| <img src=".readme_extras/demo_real.gif" alt="Demonstration" width="380" height="400">|
|:-:|:-:|
| Simulation | Real world |


## 0. Setup
 *Steps for Native (or in virtual env) setup*

0.1 (Optional) Make your venv.

```
sudo apt update
sudo apt install python3.8-venv
cd to/your/project/path
python -m venv .venv_duckietown/  #create venv
source .venv_duckietown/bin/activate  #activate it
```
 
0.2. Get the repository and step into it.

```
git clone https://github.com/TothAron/duckietown-RL-transformer-thesis.git
cd duckietown-RL-transformer-thesis/
```

0.3. Install dependecies 

```
pip install -r requirements.txt
```

## 1. Training
1.0. (Optional) Make sure you activate venv and got to the right path, if not:
```
cd to/your/project/path
source .venv_duckietown/bin/activate  #activate it
cd duckietown-RL-transformer-thesis
```

<br/>

1.1. Login to your Weights and Biases account (if do not have one, create at: https://wandb.ai)

```
wandb login # paste your token from wandb.ai/authorize
```

<br/>

1.2. Set your settings and hyperparameters in train config file at:

```
dev_and_test/config/default.py
```

<br/>

1.3. Start training. :airplane:
```
CUDA_VISIBLE_DEVICES=<gpu_idx> xvfb-run -a -s "-screen 0 1400x900x24" taskset --cpu-list <start_cpu_idx>-<end_cpu_idx> python dev_and_test/run_train.py
```
*NOTE: `<gpu_idx>` and `<start_cpu_idx>` can be selected with the help of `nvidia-smi` and `htop` unix commands relatively.*

<br/>

1.4. The only thing is left to check your training logs live at [wandb.ai](wandb.ai)
## 2. Testing

2.1. Setup your testing configuration at:
```
dev_and_test/config/test_config.py
```

<br/>

2.2. Start testing :eyeglasses:
```
CUDA_VISIBLE_DEVICES=<gpu_idx> xvfb-run -a -s "-screen 0 1400x900x24" taskset --cpu-list <start_cpu_idx>-<end_cpu_idx> python dev_and_test/run_test.py
```

<br/>

2.3. Check evaluation results at:
```
evaluation_results/ 
```

<hr />

**The used environment wrapper modules and the evaluator for testing are the modified versions of:**

```
https://github.com/kaland313/Duckietown-RL
MIT License Copyright (c) 2019 András Kalapos
```
