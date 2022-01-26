import numpy as np
import torch
import quantecon as qe
from quantecon.markov import DiscreteDP
from interpolation import interp


class DPSolver:
    def __init__(self, env):
        self.env = env
        self.result = self.solve()
        
    def solve(self):
        ddp = DiscreteDP(self.env.R, self.env.Q, self.env.BETA)
        result = ddp.solve(method='policy_iteration')
        return result

