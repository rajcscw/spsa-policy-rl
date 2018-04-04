import numpy as np
from datetime import datetime
import os
import yaml
from components.models import Weights
from components.utility import run_esn_policy_optimization_spsa, rolling_mean, plot_learning_curve
import pandas as pd

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
    Weights.ALTERNATING_SPSA,
    Weights.ALL_SPSA
]

# dataframe to hold everything
episodic_spectral_radius_strategy = pd.DataFrame()

for k in range(int(config["log"]["runs"])):

    print("-------Running iteration----------"+ str(k+1))

    # SPSA optimization
    for weight_selection in weight_selections:

        # do a run
        episodic_spectral_radius,_ = run_esn_policy_optimization_spsa(config=config,
                                                                      state_dim=state_dim,
                                                                      weight_selection=weight_selection,
                                                                      loss_function="gaussian",
                                                                      return_rad=True)

        # Compute the running mean
        N = int(config["log"]["average_every"])
        selected = rolling_mean(episodic_spectral_radius, N).tolist()

        # Combine all runs and add them to the strategy dataframe
        df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Spectral Radius": selected})
        df["run"] = k
        df["strategy"] = str(weight_selection.name).lower()
        episodic_spectral_radius_strategy = episodic_spectral_radius_strategy.append(df)

# Plot the learning curves here
folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Spectral-Radius-Comparison-all-"+str(datetime.now())
os.mkdir(folder_name)
episodic_spectral_radius_strategy.to_pickle(folder_name + "/learning_curve_df")
plot_learning_curve(folder_name +"/" + config["environment"]["name"] +"_Spectral_Radius.pdf", "Mountain Car - Continuous", episodic_spectral_radius_strategy)

# Log all the config parameters
file = open(folder_name+"/model_config.txt", "w")
file.write(str(config))
file.close()