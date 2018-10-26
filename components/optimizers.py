import numpy as np

class SPSA:
    """
    An optimizer class that implements Simultaneous Perturbation Stochastic Approximation (SPSA)
    """
    def __init__(self, a, c, A, alpha, gamma, loss_function, param_decay=0.9, exploit_p=0.95):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.param_decay = param_decay
        self.loss_function = loss_function

        # counters
        self.t = 0
        
        # keep a history of gradients
        self.gradient_history = None

        # exploring probability
        self.exploit_p = exploit_p
        
    def step(self, current_estimate, *args):
        """
        :param current_estimate: This is the current estimate of the parameter vector
        :return: returns the updated estimate of the vector
        """

        # get the current values for gain sequences
        a_t = self.a / (self.t + 1 + self.A)**self.alpha
        c_t = self.c / (self.t + 1)**self.gamma

        #print("The parameters a_t and c_t are: "+str(a_t)+","+str(c_t))

        # get the random perturbation vector from bernoulli distribution
        # it has to be symmetric around zero
        # But normal distribution does not work which makes the perturbations close to zero
        # Also, uniform distribution should not be used since they are not around zero
        if self.gradient_history is not None and np.random.rand() < self.exploit_p:
            delta = np.ones_like(self.gradient_history)
            delta[self.gradient_history <= 0.0] = -1.0
        else:
            delta = np.random.randint(0,2, current_estimate.shape) * 2 - 1

        # measure the loss function at perturbations
        loss_plus,_ = self.loss_function(current_estimate + delta * c_t, args)
        loss_minus,_ = self.loss_function(current_estimate - delta * c_t, args)

        # compute the estimate of the gradient
        g_t = (loss_plus - loss_minus) / (2.0 * delta * c_t)
        
        # add to the history with a decay parameter
        if self.gradient_history is None: 
            self.gradient_history = g_t
        else: 
            self.gradient_history = self.param_decay * self.gradient_history + (1.0 - self.param_decay) * g_t

        # update the estimate of the parameter
        current_estimate = current_estimate + a_t * g_t

        # increment the counter
        if loss_plus != loss_minus:
            self.t += 1

        return current_estimate
