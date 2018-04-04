import numpy as np
from datetime import datetime
import os
import yaml
from components.models import Weights
from components.utility import run_esn_policy_optimization_spsa, rolling_mean, plot_learning_curve
import pandas as pd
from baseline_components.utility import run_actor_critic

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


weight_selections = [
    Weights.OUTPUT_SPSA,
    Weights.ALTERNATING_SPSA,
    Weights.ALL_SPSA
]

# dataframe to hold everything
episodic_total_rewards_strategy = pd.DataFrame()

for k in range(int(config["log"]["runs"])):

    print("-------Running iteration----------"+ str(k+1))

    # SPSA optimization
    for weight_selection in weight_selections:

        # do a run
        episodic_total_reward,_ = run_esn_policy_optimization_spsa(config, state_dim, weight_selection, "gaussian")

        # Compute the running mean
        N = int(config["log"]["average_every"])
        selected = rolling_mean(episodic_total_reward, N, -90).tolist()

        # Combine all runs and add them to the strategy dataframe
        if selected[-1] > 0: # add only if the training succeeded
            df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": selected})
            df["run"] = k
            df["strategy"] = str(weight_selection.name).lower()
            episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

    # Baseline - Actor-critic
    episodic_total_reward = run_actor_critic(config)

    # Compute the running mean
    N = int(config["log"]["average_every"])
    selected = rolling_mean(episodic_total_reward, N, -90).tolist()

    if selected[-1] > 0: # add only if the training succeeded
        # Combine all runs and add them to the strategy dataframe
        df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": selected})
        df["run"] = k
        df["strategy"] = "Actor-Critic"
        episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

# Plot the learning curves here
folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Comparison-all-"+str(datetime.now())
os.mkdir(folder_name)
episodic_total_rewards_strategy.to_pickle(folder_name+"/learning_curve_df")
plot_learning_curve(folder_name+"/"+config["environment"]["name"]+"_Episodic_total_reward.pdf", "Mountain Car - Continuous", episodic_total_rewards_strategy)


# Get the average total reward for each episode (combining all runs)
threshold = episodic_total_rewards_strategy["Iteration"].max() - int(config["log"]["final_average_per_strategy"])
episodic_total_rewards_strategy = episodic_total_rewards_strategy[episodic_total_rewards_strategy["Iteration"] > threshold]
grouped = episodic_total_rewards_strategy[["strategy", "Episodic Total Reward"]].groupby(["strategy"]).mean()
file = open(folder_name+"/strategy_average.txt", "w")
file.write(grouped.to_string())
file.close()


# Log all the config parameters
file = open(folder_name+"/model_config.txt", "w")
file.write(str(config))
file.close()