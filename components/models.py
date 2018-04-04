import numpy as np
from components import activations
import scipy.linalg as la
from enum import Enum


class Weights(Enum):
    OUTPUT_SPSA = 1
    RESERVOIR_SPSA = 2
    INPUT_SPSA = 3
    ALL_SPSA = 4
    ALTERNATING_SPSA = 5


class LinearModel:
    def __init__(self, input_d, output_d, init, reg, optimizer):
        self.input_d = input_d
        self.output_d = output_d
        self.reg = reg
        self.optimizer = optimizer

        # initialize the weights
        self.W = init

    def forward(self, input, update=True):
        return np.dot(self.W, input)

    def set_parameter(self, parameter):
        self.W = parameter

    def get_parameter(self):
        return self.W

    def reset(self):
        return True

    def update(self):
        return True


class EchoStateNetwork:
    def __init__(self,
                 size,
                 input_d,
                 output_d,
                 spectral_radius,
                 leaking_rate,
                 initial_transient,
                 input_weight=None,
                 reservoir_weight=None,
                 output_weight=None,
                 optimize_weights=Weights.OUTPUT_SPSA,
                 reservoir_activation_function=activations.HyperbolicTangent(),
                 output_activation_function=activations.Linear(),
                 sgd_lr=1e-4,
                 optimizer=None):
        self.n_r = size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.initial_transient = initial_transient

        # Initialize weights
        self.n_i = input_d
        self.n_o = output_d
        self.W_i = np.copy(input_weight)
        self.W_r = np.copy(reservoir_weight)

        # linear readout
        self.linear_model = LinearModel(input_d=self.n_r,
                                        output_d=output_d,
                                        init=np.copy(output_weight),
                                        reg=sgd_lr,
                                        optimizer=optimizer)

        self.__force_spectral_radius()

        # Internal states
        self.latest_r = np.zeros((self.n_r,1))

        # Activation functions
        self.reservoir_activation = reservoir_activation_function
        self.output_activation = output_activation_function

        # transient counters
        self.transient_counter = 0

        # weight selection
        self.weight_selection = optimize_weights

        # to store the current matrix that is being optimized (in case of alternating)
        if self.weight_selection == Weights.ALTERNATING_SPSA:
            self.current_optimized = Weights.INPUT_SPSA

    def __force_spectral_radius(self):
        # Make the reservoir weight matrix - a unit spectral radius
        rad = np.max(np.abs(la.eigvals(self.W_r)))
        self.W_r = self.W_r / rad

        # Force spectral radius
        self.W_r = self.W_r * self.spectral_radius

    def forward(self, input, update=True):
        # Reservoir state
        term1 = np.dot(self.W_i, input)
        term2 = np.dot(self.W_r, self.latest_r)
        r_t = (1.0 - self.leaking_rate) * self.latest_r + self.leaking_rate * self.reservoir_activation(term1 + term2)

        # Output
        output = self.output_activation(self.linear_model.forward(r_t))

        if update:
            self.transient_counter += 1
            self.latest_r = r_t

        return output, r_t

    def reset(self):
        self.latest_r = np.zeros((self.n_r, 1))
        self.transient_counter = 0

    def get_spectral_radius(self):
        return np.max(np.abs(la.eigvals(self.W_r)))

    def get_parameter(self):
        # here we need to fetch weights based on the weight selection
        if self.weight_selection == Weights.OUTPUT_SPSA:
            return self.linear_model.get_parameter()
        elif self.weight_selection == Weights.RESERVOIR_SPSA:
            return self.W_r
        elif self.weight_selection == Weights.INPUT_SPSA:
            return self.W_i
        elif self.weight_selection == Weights.ALL_SPSA:
            stacked = np.hstack((self.W_i.flatten(), self.W_r.flatten(), self.linear_model.get_parameter().flatten()))
            stacked = stacked.reshape((-1,1))
            return stacked
        elif self.weight_selection == Weights.ALTERNATING_SPSA:
            # in case of alternating, always get what is being optimized
            if self.current_optimized == Weights.INPUT_SPSA:
                return self.W_i
            elif self.current_optimized == Weights.RESERVOIR_SPSA:
                return self.W_r
            elif self.current_optimized == Weights.OUTPUT_SPSA:
                return self.linear_model.get_parameter()
            else:
                return None

    def set_parameter(self, parameter):
        # here we need to set weights based on the weight selection
        if self.weight_selection == Weights.OUTPUT_SPSA:
            self.linear_model.set_parameter(parameter)
        elif self.weight_selection == Weights.RESERVOIR_SPSA:
            self.W_r = parameter
        elif self.weight_selection == Weights.INPUT_SPSA:
            self.W_i = parameter
        elif self.weight_selection == Weights.ALL_SPSA:
            # Split the parameter to obtain input, reservoir and output matrices
            input_size = self.W_i.size
            res_size = self.W_r.size
            input_weight, reservoir_weight, output_weight = \
            np.split(parameter, [input_size, input_size+res_size])

            # Set the parameters
            self.W_i = input_weight.reshape(self.W_i.shape)
            self.W_r = reservoir_weight.reshape(self.W_r.shape)
            self.linear_model.set_parameter(output_weight.reshape(self.linear_model.W.shape))

        elif self.weight_selection == Weights.ALTERNATING_SPSA:
            # in case of alternating, set must be synchronized what is being optimized
            if self.current_optimized == Weights.INPUT_SPSA:
                self.W_i = parameter
            elif self.current_optimized == Weights.RESERVOIR_SPSA:
                self.W_r = parameter
            elif self.current_optimized == Weights.OUTPUT_SPSA:
                self.linear_model.set_parameter(parameter)

    def alternate(self):
        if self.weight_selection == Weights.ALTERNATING_SPSA:
            if self.current_optimized == Weights.INPUT_SPSA:
                self.current_optimized = Weights.RESERVOIR_SPSA
            elif self.current_optimized == Weights.RESERVOIR_SPSA:
                self.current_optimized = Weights.OUTPUT_SPSA
            elif self.current_optimized == Weights.OUTPUT_SPSA:
                self.current_optimized = Weights.INPUT_SPSA
