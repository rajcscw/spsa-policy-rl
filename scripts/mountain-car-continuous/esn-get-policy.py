import numpy as np
from datetime import datetime
import os
import yaml
from components.models import Weights
from components.utility import run_esn_policy_optimization_spsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# read the config
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+"/config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

# Just convert state indices
if len(str(config["states"]["partial"])) == 1:
    config["environment"]["state_select"] = np.array([int(config["states"]["partial"])])
else:
    config["environment"]["state_select"] = np.array(config["states"]["partial"].split(",")).astype(int)

# Other variables
max_iter = int(config["log"]["iterations"])
state_dim = config["environment"]["state_select"].shape[0]
action_dim = config["environment"]["action_space"]


folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Policy-Vis-"+str(datetime.now())
os.mkdir(folder_name)

# weight selection
weight_selection = Weights.ALL_SPSA

# dataframe to hold everything
episodic_total_rewards_strategy = pd.DataFrame()

# do a run
episodic_total_reward, chosen_actions = run_esn_policy_optimization_spsa(config, state_dim, weight_selection, "gaussian", None, True)
print(chosen_actions)

# Log all the config parameters
file = open(folder_name+"/model_config.txt", "w")
file.write(str(config))
file.close()

# Visualize the distribution
sns.set()
chosen_actions = np.array(chosen_actions).reshape(1,-1)
kde = stats.gaussian_kde(chosen_actions)
x = np.linspace(-2,2, 100)
pdf = kde(x)
sum = np.sum(pdf)
pdf = pdf / sum
print(pdf.sum())
plt.plot(x.flatten(), pdf.flatten())
plt.xlabel("action")
plt.ylabel("p(action)")
plt.fill_between(x.flatten(), pdf.flatten(), color="r", alpha=0.5)
plt.savefig(folder_name+"/Policy.pdf")