import numpy as np
import torch
from interpolation import interp


class DPSolver:

    def __init__(self, cfg):

        self.beta, self.gamma = cfg.BETA, cfg.GAMMA
        self.x_grid = np.linspace(cfg.ENV_STATE_SPACE_L, cfg.ENV_STATE_SPACE_H, 100)
        self.x_grid_t = torch.tensor(self.x_grid).unsqueeze(-1)
        self.v_array = np.zeros(len(self.x_grid)) # Initial guess
    # Utility function
    def u(self, c):
        if self.gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - self.gamma)) / (1 - self.gamma)

    # first derivative of utility function
    def u_prime(self, c):
        return c ** (-self.gamma)

    def state_action_value(self, actor, v_array):
        """
        Right hand side of the Bellman equation given x and c.
        Notice that actor is an actor that only take torch tensor as input
        """

        u, beta = self.u, self.beta
        v = lambda x: interp(self.x_grid, v_array, x)
        c = actor(self.x_grid_t).squeeze(-1).detach().numpy()
        return u(c) + beta * v(self.x_grid - c)

    def solve(self, actor, tol=1e-4, max_iter=1000):
        v_array = self.v_array
        i = 0
        error = tol + 1
        # iteration
        while i < max_iter and error > tol:
            v_new_array = self.state_action_value(actor, v_array)
            error = np.max(np.abs(v_array - v_new_array))
            i += 1

            v_array = v_new_array
        # check convergence
        if i == max_iter:
            print("Failed to converge!")
        else:
            print(f"\nConverged in {i} iterations.")

        self.v_array = v_array

    def __call__(self, s_t, a_t):
        # input a batch (s, a) tensors and output Q(s, a) as a linear interpolation of solved v_array
        next_s = (s_t - a_t).squeeze(-1).numpy()
        c = a_t.squeeze(-1).numpy()
        v = lambda x: interp(self.x_grid, self.v_array, x)
        return torch.tensor(self.u(c) + self.beta * v(next_s), dtype=torch.float32).unsqueeze(-1)
