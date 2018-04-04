import numpy as np
import gym
from sklearn.preprocessing import StandardScaler
from gym import wrappers, logger


class FixedIntervalVideoSchedule(object):
    def __init__(self, iterations, interval):
        self.sequence = []
        last = -1
        for i in range(iterations):
            last = last + 3
            if (i + 1) % interval == 0 or i == 0:
                self.sequence.append(last)
        self.sequence = np.array(self.sequence)

    def __call__(self, count):
        return count in self.sequence


class EpisodicReturnSoftmaxPolicy(object):
    def __init__(self, model, config, save_loc):
        self.config = config

        # Set up the environment
        self.env_params = self.config["environment"]
        self.env = gym.make(self.env_params["name"])
        if self.env_params["max_episode_steps"] != "none":
            self.env._max_episode_steps = int(self.env_params["max_episode_steps"])

        if save_loc is not None:
            self.env = wrappers.Monitor(self.env,
                                        directory=save_loc,
                                        force=True,
                                        video_callable=FixedIntervalVideoSchedule(iterations=config["log"]["iterations"],
                                                                                  interval=config["log"]["interval"]))

        # Get the dimensions of state representation
        self.n_sv = self.config["environment"]["state_select"].shape[0]

        # Get the dimensions of action space
        self.n_a = self.env.action_space.n

        # Set up the model
        self.model = model

        # Scale the inputs
        self.scaler = StandardScaler()
        if self.scaler is not None:
            observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
            self.scaler.fit(observation_examples)

    def policy(self, probs):
        probs = probs.flatten().tolist()
        chosen_action = np.random.choice(np.arange(len(probs)), p=probs)
        return chosen_action

    def get_best_action(self,state, update=True):
        # transform
        if self.scaler is not None:
            state = self.scaler.transform(state.reshape((1,-1))).flatten()
        state = state[self.env_params["state_select"]].reshape(self.n_sv,1)
        probs, _ = self.model.forward(state, update=update)
        chosen_action = self.policy(probs)
        return probs, chosen_action

    def __call__(self, parameter, *args):
        """
        This functions plays an episode in the open AI gym environment
        :return: Returns episodic return
        """

        log = args[0]

        # set up the model with the parameter
        self.model.set_parameter(parameter)

        # Play an episode
        current_state = self.env.reset()
        episodic_return = 0
        while True:
            # Render the environment
            if log:
                self.env.render()

            # Choose the best action
            current_estimates, current_action = self.get_best_action(current_state)

            # Perform the chosen action and get the reward and next state of the environment
            next_state, current_reward, done, info = self.env.step(current_action)
            episodic_return += current_reward

            if done:
                break

            # Set the next state
            current_state = next_state

        # reset model
        self.model.reset()

        episodic_return = episodic_return + np.random.normal(scale=float(self.config["SPSA"]["noise"]), size=1)

        return episodic_return, episodic_return


class EpisodicReturnGaussianPolicy(object):
    def __init__(self, model, config, save_loc):
        self.config = config

        # Set up the environment
        self.env_params = self.config["environment"]
        self.env = gym.make(self.env_params["name"])
        if self.env_params["max_episode_steps"] != "none":
            self.env._max_episode_steps = int(self.env_params["max_episode_steps"])

        if save_loc is not None:
            self.env = wrappers.Monitor(self.env,
                                        directory=save_loc,
                                        force=True,
                                        video_callable=FixedIntervalVideoSchedule(iterations=config["log"]["iterations"],
                                                                                  interval=config["log"]["interval"]))

        # Get the dimensions of state representation
        self.n_sv = self.config["environment"]["state_select"].shape[0]

        # Get the dimensions of action space
        self.n_a = self.config["environment"]["action_space"]

        # Set up the model
        self.model = model

        # Scale the inputs
        self.scaler = StandardScaler()
        if self.scaler is not None:
            observation_examples = np.array([self.env.observation_space.sample() for x in range(20000)])
            self.scaler.fit(observation_examples)

    def get_best_action(self,state, update=True):

        # transform states
        if self.scaler is not None:
            state = self.scaler.transform(state.reshape((1,-1))).flatten()

        # choose the action
        state = state[self.env_params["state_select"]].reshape(self.n_sv,1)
        network_output, _ = self.model.forward(state, update=update)

        # apply gaussian policy
        action = self.policy(network_output)

        # apply the bounds
        a_min = self.env.action_space.low.reshape((-1,1))
        a_max = self.env.action_space.high.reshape((-1,1))
        action = np.clip(action, a_min, a_max)

        return action

    def policy(self, network_output):

        # break them into mean (mu) and sigma of gaussian distribution
        mean, sd = network_output[:self.n_a], network_output[self.n_a:]

        sd = np.log(1 + np.exp(sd))

        action = np.random.normal(loc=mean, scale=sd, size=(self.n_a,1))
        return action

    def __call__(self, parameter, *args):
        """
        This functions plays an episode in the open AI gym environment
        :return: Returns episodic return
        """
        log = args[0]

        # set up the model with the parameter
        self.model.set_parameter(parameter)

        # Play an episode
        current_state = self.env.reset()
        episodic_return = 0
        chosen_actions = []
        while True:
            # Render the environment
            if log:
                self.env.render()

            # Choose the best action
            current_action = self.get_best_action(current_state)

            # log the chosen actions if states satisfy selection criteria
            if self.config["PolicyVis"]:
                state_start = np.array(self.config["PolicyVis"]["selected_state_start"].split(",")).astype(float)
                state_end = np.array(self.config["PolicyVis"]["selected_state_end"].split(",")).astype(float)
                if np.all(state_start <= current_state.flatten()) and np.all(current_state.flatten() <= state_end):
                    chosen_actions.append(current_action)

            # Perform the chosen action and get the reward and next state of the environment
            next_state, current_reward, done, info = self.env.step(current_action)
            episodic_return += current_reward

            if done:
                break

            # Set the next state
            current_state = next_state

        # reset model
        self.model.reset()

        episodic_return = episodic_return + np.random.normal(scale=float(self.config["SPSA"]["noise"]), size=1)

        return episodic_return, chosen_actions
