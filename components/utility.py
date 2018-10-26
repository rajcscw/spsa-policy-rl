import numpy as np
from components import topologies
from components.models import EchoStateNetwork, Weights
from components.activations import SoftMax
from components.loss import EpisodicReturnSoftmaxPolicy, EpisodicReturnGaussianPolicy
from components.optimizers import SPSA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def run_esn_policy_optimization_spsa(config, state_dim, weight_selection, exploit_p, loss_function, save_loc=None, log_policy=False, return_rad=False):
    """
    
    :param config: configuration parameters
    :param state_dim: state dimension
    :param action_dim: action dimension
    :return: episodic_total_reward
    """

    # action dimensions depending on the type of task
    if loss_function == "softmax": # TBD: works only for single action with multiple choices
        action_dim = int(config["environment"]["action_space"])
    elif loss_function == "gaussian":
        action_dim = int(config["environment"]["action_space"]) * 2 # for mean and variance

    # Initialize all the weights
    input_weight = topologies.RandomInputTopology(inputSize=state_dim,
                                                  reservoirSize=config["ESN"]["res_size"],
                                                  inputConnectivity=config["ESN"]["input_conn"]). \
        generateWeightMatrix(scaling=float(config["ESN"]["input_scaling"]))

    reservoir_weight = topologies.RandomReservoirTopology(size=config["ESN"]["res_size"],
                                                          connectivity=config["ESN"]["res_conn"]). \
        generateWeightMatrix(scaling=float(config["ESN"]["res_scaling"]))

    output_weight = np.random.uniform(low=-float(config["ESN"]["res_scaling"]), high=float(config["ESN"]["res_scaling"]), size=(action_dim,config["ESN"]["res_size"]))

    model = EchoStateNetwork(size=config["ESN"]["res_size"],
                             input_d=config["environment"]["state_select"].shape[0],
                             output_d=int(config["environment"]["action_space"]),
                             spectral_radius=config["ESN"]["spectral_radius"],
                             leaking_rate=config["ESN"]["leaking_rate"],
                             initial_transient=config["ESN"]["initial_transient"],
                             input_weight=input_weight,
                             reservoir_weight=reservoir_weight,
                             output_weight=output_weight,
                             optimize_weights=weight_selection,
                             output_activation_function=SoftMax())

    # learning rate adjustment
    # if only output weights, then just update your weights faster
    # so we need two separate learning rates
    if weight_selection == Weights.OUTPUT_SPSA:
        spsa_a = float(config["SPSA"]["a_output"])
    elif weight_selection == Weights.ALTERNATING_SPSA:
        spsa_a = float(config["SPSA"]["a_alter"])
    else:
        spsa_a = float(config["SPSA"]["a_all"])

    # objective function
    if loss_function == "softmax":
        objective = EpisodicReturnSoftmaxPolicy(model=model, config=config, save_loc=save_loc)
    elif loss_function == "gaussian":
        objective = EpisodicReturnGaussianPolicy(model=model, config=config, save_loc=save_loc)

    # Optimizer
    optimizer = SPSA(a=spsa_a,
                     c=float(config["SPSA"]["c"]),
                     A=float(config["SPSA"]["A"]),
                     alpha=float(config["SPSA"]["alpha"]),
                     gamma=float(config["SPSA"]["gamma"]),
                     param_decay=float(config["SPSA"]["decay"]),
                     exploit_p=exploit_p,
                     loss_function=objective)

    # the main loop
    episodic_total_reward = []
    episodic_spectral_radius = []
    for i in range(config["log"]["iterations"]):
        current_estimate = model.get_parameter()
        new_estimate = optimizer.step(current_estimate)
        model.set_parameter(new_estimate)

        # evaluate the learning
        total_reward, _ = objective(model.get_parameter(), False)

        # get the spectral radius
        #rad = model.get_spectral_radius()
        rad = 0.0

        print("Evaluating at iteration:"+str(i)+", episodic return:"+str(total_reward) + ", spectral radius:"+str(rad))

        # book keeping stuff
        episodic_total_reward.append(total_reward)
        episodic_spectral_radius.append(rad)

        model.alternate()

    # now that the learning is complete, we can log the policy by running few episodes
    # and get the state and action pairs
    chosen_actions = []

    if log_policy:
        for i in range(config["PolicyVis"]["episodes"]):
            _, actions = objective(model.get_parameter(), False)
            chosen_actions.extend(actions)

    if return_rad:
        return episodic_spectral_radius, chosen_actions

    return episodic_total_reward, chosen_actions


def rolling_mean(X, window_size, pad=None):
    # pad in the front
    if pad is None:
        front = np.full((window_size,), X[0]).tolist()
    else:
        front = np.full((window_size,), pad).tolist()
    padded = front + X
    mean = np.convolve(padded, np.ones((window_size,))/window_size, "valid")
    return mean


def plot_learning_curve(file, title, series, value="Episodic Total Reward"):
    sns.set(style="darkgrid")
    sns.set_context("paper")
    plt.title(title, fontsize=12)
    sns.tsplot(data=series, time="Iteration", unit="run", condition="strategy", value=value)
    plt.legend(loc="lower right", fontsize=12)
    plt.ylabel(value, fontsize=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.savefig(file)