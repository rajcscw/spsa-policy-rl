import numpy as np
from components import activations
import scipy.linalg as la
import tensorflow as tf


class GaussianPolicyEstimator():
    def __init__(self, state_dim, action_bounds, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [state_dim], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # a linear regressor for fitting gaussian parameters
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, action_bounds[0], action_bounds[1])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # Session initialize
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

    def forward(self, state):
        return self.sess.run(self.action, { self.state: state })

    def backward(self, params):
        state, target, action = params
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


class SoftmaxPolicyEstimator():
    def __init__(self, state_dim, action_dim, learning_rate=0.01, scope="policy_estimator", optimizer="rmsprop", reg_factor=0.0):
        with tf.variable_scope(scope):
            self.action_dim = action_dim
            self.state = tf.placeholder(tf.float32, [state_dim], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=action_dim,
                activation_fn=None,
                weights_initializer=tf.keras.initializers.glorot_normal())

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.loss += reg_factor * tf.nn.l2_loss(self.output_layer)

            if optimizer == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer == "rmsprop":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # Session initialize
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

    def forward(self, state):
        action_probs = self.sess.run(self.action_probs, { self.state: state })
        action = np.random.choice(np.arange(self.action_dim), p=action_probs)
        return action

    def backward(self, params):
        state, target, action = params
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    def __init__(self, state_dim, learning_rate=0.1, scope="value_estimator", optimizer="adam", reg_factor=0.0):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [state_dim], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # a linear regressor for fitting value functions
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.keras.initializers.glorot_normal(),
                weights_regularizer=tf.contrib.layers.l2_regularizer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)
            self.loss += reg_factor * tf.nn.l2_loss(self.output_layer)

            if optimizer == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer == "rmsprop":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # Session initialize
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

    def forward(self, state):
        return self.sess.run(self.value_estimate, { self.state: state })

    def backward(self, params):
        state, target = params
        feed_dict = { self.state: state, self.target: target }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss


class EchoStateNetwork:
    def __init__(self,
                 size,
                 input_d,
                 output_d,
                 spectral_radius,
                 leaking_rate,
                 initial_transient,
                 estimator,
                 state_select,
                 scaler,
                 input_weight=None,
                 reservoir_weight=None,
                 reservoir_activation_function=activations.HyperbolicTangent()):
        self.n_r = size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.initial_transient = initial_transient
        self.state_select = state_select
        self.scaler = scaler

        # Initialize weights
        self.n_i = input_d
        self.n_o = output_d
        self.W_i = np.copy(input_weight)
        self.W_r = np.copy(reservoir_weight)

        self.__force_spectral_radius()

        # Internal states
        self.latest_r = np.zeros((self.n_r,1))

        # Activation functions
        self.reservoir_activation = reservoir_activation_function

        # transient counters
        self.transient_counter = 0

        # set up the estimators for policy and value functions
        self.estimator = estimator

        self.warmed_up = False

    def __force_spectral_radius(self):
        # Make the reservoir weight matrix - a unit spectral radius
        rad = np.max(np.abs(la.eigvals(self.W_r)))
        self.W_r = self.W_r / rad

        # Force spectral radius
        self.W_r = self.W_r * self.spectral_radius

    def warm_up(self, input):
        for i in range(self.initial_transient):
            term1 = np.dot(self.W_i, input)
            term2 = np.dot(self.W_r, self.latest_r)
            r_t = (1.0 - self.leaking_rate) * self.latest_r + self.leaking_rate * self.reservoir_activation(term1 + term2)
            self.latest_r = r_t

    def forward(self, input, update=True):
        # scaler
        input = self.scaler.transform(input.reshape(1, -1)).flatten()

        # state select (because of partial and full observability)
        input = input[self.state_select].reshape((-1, 1))

        # warm up
        if not self.warmed_up:
            self.warm_up(input)
            self.warmed_up = True

        term1 = np.dot(self.W_i, input)
        term2 = np.dot(self.W_r, self.latest_r)
        r_t = (1.0 - self.leaking_rate) * self.latest_r + self.leaking_rate * self.reservoir_activation(term1 + term2)

        # Output
        output = self.estimator.forward(r_t.flatten())

        if update:
            self.transient_counter += 1
            self.latest_r = r_t

        return output, r_t

    def backward(self, params):
        return self.estimator.backward(params)

    def reset(self):
        self.latest_r = np.zeros((self.n_r, 1))
        self.transient_counter = 0
        self.warmed_up = False