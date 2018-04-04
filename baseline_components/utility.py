import numpy as np
import tensorflow as tf
from components import topologies
import gym
from baseline_components.estimators import EchoStateNetwork, GaussianPolicyEstimator, SoftmaxPolicyEstimator, ValueEstimator
from baseline_components.agents import ActorCriticAgent
import sklearn


def run_actor_critic(config, policy="gaussian"):
    # create the environment
    env = gym.envs.make(config["environment"]["name"])

    # Just convert state indices
    if len(str(config["states"]["partial"])) == 1:
        config["environment"]["state_select"] = np.array([int(config["states"]["partial"])])
    else:
        config["environment"]["state_select"] = np.array(config["states"]["partial"].split(",")).astype(int)

    state_dim = config["environment"]["state_select"].shape[0]

    # scaler
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # create the ESNs for policy and value estimators
    input_weight = topologies.RandomInputTopology(inputSize=state_dim,
                                                  reservoirSize=config["ESN"]["res_size"],
                                                  inputConnectivity=config["ESN"]["input_conn"]). \
        generateWeightMatrix(scaling=float(config["ESN"]["input_scaling"]))

    reservoir_weight = topologies.RandomReservoirTopology(size=config["ESN"]["res_size"],
                                                          connectivity=config["ESN"]["res_conn"]). \
        generateWeightMatrix(scaling=float(config["ESN"]["res_scaling"]))

    if policy == "gaussian":
        policy_estimator = GaussianPolicyEstimator(state_dim=config["ESN"]["res_size"],
                                                   action_bounds=(env.action_space.low[0], env.action_space.high[0]),
                                                   learning_rate=float(config["PolicyEstimator"]["lr"]))
    elif policy == "softmax":
        policy_estimator = SoftmaxPolicyEstimator(state_dim=config["ESN"]["res_size"],
                                                  action_dim=int(config["environment"]["action_space"]),
                                                  learning_rate=float(config["PolicyEstimator"]["lr"]))

    # tf variables
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    policy_network = EchoStateNetwork(size=config["ESN"]["res_size"],
                                      input_d=config["environment"]["state_select"].shape[0],
                                      output_d=int(config["environment"]["action_space"]),
                                      spectral_radius=config["ESN"]["spectral_radius"],
                                      leaking_rate=config["ESN"]["leaking_rate"],
                                      initial_transient=config["ESN"]["initial_transient"],
                                      input_weight=input_weight,
                                      reservoir_weight=reservoir_weight,
                                      state_select=config["environment"]["state_select"],
                                      scaler=scaler,
                                      estimator=policy_estimator)

    value_network = EchoStateNetwork(size=config["ESN"]["res_size"],
                                     input_d=config["environment"]["state_select"].shape[0],
                                     output_d=int(config["environment"]["action_space"]),
                                     spectral_radius=config["ESN"]["spectral_radius"],
                                     leaking_rate=config["ESN"]["leaking_rate"],
                                     initial_transient=config["ESN"]["initial_transient"],
                                     input_weight=input_weight,
                                     reservoir_weight=reservoir_weight,
                                     state_select =config["environment"]["state_select"],
                                     scaler=scaler,
                                     estimator=ValueEstimator(state_dim=config["ESN"]["res_size"],
                                                              learning_rate=float(config["ValueEstimator"]["lr"])))

    # create the agent
    agent = ActorCriticAgent(env, policy_network, value_network, config["MDP"]["discount_factor"])
    episodic_total_reward = agent.run(config["log"]["iterations"])
    return episodic_total_reward