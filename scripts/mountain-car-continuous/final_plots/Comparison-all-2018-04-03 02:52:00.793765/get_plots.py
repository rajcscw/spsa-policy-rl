import pandas as pd
import pickle
import yaml
from components.utility import plot_learning_curve

# read the config
with open("config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

# read the dataframe
episodic_total_rewards_strategy = pd.read_pickle("learning_curve_df")

# just change all strategy values to lower case
episodic_total_rewards_strategy["strategy"] = episodic_total_rewards_strategy["strategy"].apply(lambda x: x.lower())

# Plot the learning curves here
plot_learning_curve(config["environment"]["name"]+"_Episodic_total_reward.pdf", "Mountain Car - Continuous", episodic_total_rewards_strategy)

# Get the average total reward for each episode (combining all runs)
threshold = episodic_total_rewards_strategy["Iteration"].max() - int(config["log"]["final_average_per_strategy"])
episodic_total_rewards_strategy = episodic_total_rewards_strategy[episodic_total_rewards_strategy["Iteration"] > threshold]
grouped = episodic_total_rewards_strategy[["strategy", "Episodic Total Reward"]].groupby(["strategy"]).mean()
file = open("strategy_mean_2.txt", "w")
file.write(grouped.to_string())
file.close()

grouped = episodic_total_rewards_strategy[["strategy", "Episodic Total Reward"]].groupby(["strategy"]).std()
file = open("strategy_std.txt", "w")
file.write(grouped.to_string())
file.close()