import gym
from gym.utils import seeding
import numpy as np
import torch
from numba import jit

gym.logger.set_level(40)

class EconModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.r, self.w, self.BETA = cfg.r, cfg.w, cfg.BETA
        self.a_min, self.a_max, self.a_size = float(cfg.a_min), cfg.a_max, cfg.a_size
        self.PI = np.array(cfg.PI).reshape(2, 2)
        self.z_vals = np.array(cfg.z_vals)
        self.z_size = len(self.z_vals)
        self.a_vals = np.linspace(self.a_min, self.a_max, self.a_size)
        self.n = self.a_size * self.z_size

        # Build the array Q
        self.Q = np.zeros((self.n, self.a_size, self.n))
        self.build_Q()

        # Build the array R
        self.R = np.empty((self.n, self.a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        Use this method to reset prices. Calling the method will trigger a
        re-build of R.
        """
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.PI)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R,
                self.a_size,
                self.z_size,
                self.a_vals,
                self.z_vals,
                self.r,
                self.w)


# Do the hard work using JIT-ed functions

@jit(nopython=True)
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z + (1 + r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = np.log(c)  # Utility


@jit(nopython=True)
def populate_Q(Q, a_size, z_size, Π):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i*z_size + next_z_i] = Π[z_i, next_z_i]


@jit(nopython=True)
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i*z_size + z_i]
    return a_probs


class AiyagariEnv(EconModel):
    
    def __init__(self, cfg):
        EconModel.__init__(self, cfg)

        self.observation_space = gym.spaces.Box(low=np.array([self.a_min, self.z_vals[0]], dtype=np.float32),
                                   high=np.array([self.a_max, self.z_vals[1]], dtype=np.float32), shape=(2, ))
        self.action_space = gym.spaces.Box(low=self.a_min, high=self.a_max, dtype=np.float32, shape=(1,))

        # seeding
        self.np_random, _ = seeding.np_random(cfg.seed)
        self.observation_space.seed(cfg.seed)
        self.action_space.seed(cfg.seed)

    def state_sampling(self, B):
        temp = torch.rand(B, 2)
        s = torch.zeros_like(temp)
        s[:, 0] = temp[:, 0] * self.a_max + self.a_min
        s[temp[:, 1] > 0.5, 1] = 0
        s[temp[:, 1] <= 0.5, 1] = 1
        return s

    def state_linspace(self, B):
        s = torch.zeros(B, 2)
        s[: B//2, 0] = torch.linspace(self.a_min, self.a_max, B//2)
        s[B//2:, 0] = torch.linspace(self.a_min, self.a_max, B//2)
        s[: B//2, 1] = 0
        s[B//2:, 1] = 1
        return s 

    def gen_states(self, sampling=None, B=None):

        if sampling is None:
            sampling = self.cfg.ENV_GEN_STATE_SAMPLING
        if B is None:
            B = self.cfg.BATCH_SIZE

        return self.state_sampling(B) if sampling == True else self.state_linspace(B)

    def shock_transition(self, z_idx):
        # input current shock batch z_idx
        # return simulated next shock according to the shock transition matrix
        z_idx = z_idx.long().squeeze(-1)
        p = torch.tensor(self.PI)[z_idx]  # condition probability
        next_z_idx = torch.multinomial(p, num_samples=1, replacement=True)
        return next_z_idx.float()

    def transit(self, state, action, num_steps=None):

        if num_steps is None:
            num_steps = self.cfg.ENV_TRANSITION_NUM_STEPS
        R = 0
        beta = self.cfg.BETA
        for i in range(num_steps): 
            a, z = state[:, 0:1].clone(), state[:, 1:2].clone() * 0.9 + 0.1
            a_new = action
            w = self.cfg.w
            r = self.cfg.r
            c = w * z + (1 + r) * a - a_new
            r = torch.log(c)
            state[:, 0:1] = a_new.clone()
            state[:, 1:2] = self.shock_transition(state[:, 1:2].clone())
            R += r*beta**i

        return R, state