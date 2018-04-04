class ActorCriticAgent():
    def __init__(self, env, policy_estimator, value_estimator, discount_factor):
        self.env = env
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator
        self.discount_factor = discount_factor

    def run(self, n_episodes, negative_reward=False):
        episodic_total_reward = []
        for i_episode in range(n_episodes):
            # Reset the environment
            state = self.env.reset()

            total_episodic_reward = 0
            policy_estimator_loss = 0
            value_estimator_loss = 0

            while True:
                #self.env.render()

                # Take a step
                action, policy_res_state = self.policy_estimator.forward(state)
                next_state, reward, done, _ = self.env.step(action)

                # Update statistics
                total_episodic_reward += reward

                if done and negative_reward:
                    reward = -100

                # Calculate TD Target
                current_estimate, value_res_state = self.value_estimator.forward(state)
                value_next, _ = self.value_estimator.forward(next_state, update=False)
                td_target = reward + self.discount_factor * value_next
                td_error = td_target - current_estimate

                # Update the value estimator
                value_estimator_loss += self.value_estimator.backward((value_res_state.flatten(), td_target))

                # Update the policy estimator
                # using the td error as our advantage estimate
                policy_estimator_loss += self.policy_estimator.backward((policy_res_state.flatten(), td_error, action))

                if done:
                    break

                state = next_state

            print("Episode " + str(i_episode+1)+", Total Reward: " + str(total_episodic_reward) + " Policy Total loss: " +str(policy_estimator_loss) + " Value Total loss: "+str(value_estimator_loss))
            episodic_total_reward.append(total_episodic_reward)

            # reset internal states of the model
            self.policy_estimator.reset()
            self.value_estimator.reset()

        return episodic_total_reward
