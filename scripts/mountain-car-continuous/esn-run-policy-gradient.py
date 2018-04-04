import numpy as np
import yaml
import os
import pandas as pd
import seaborn as sns
from datetime import datetime
from baseline_components.utility import run_actor_critic
from components.utility import rolling_mean, plot_learning_curve

# set plot context and color palette
sns.set_context("paper")
sns.set(style="darkgrid")

# load the configuration
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+"/config.yml") as f:
    config = yaml.load(f)

print("The configuration is: "+str(config))

episodic_total_rewards_strategy = pd.DataFrame()
for k in range(config["log"]["runs"]):
    episodic_total_reward = run_actor_critic(config)

    # Compute the running mean
    N = int(config["log"]["average_every"])
    selected = rolling_mean(episodic_total_reward, N).tolist()

    # Combine all runs and add them to the strategy dataframe
    df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": selected})
    df["run"] = k
    df["strategy"] = "Actor-Critic"
    episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)


# Plot the learning curves here
folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/PolicyGradient-"+str(datetime.now())
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