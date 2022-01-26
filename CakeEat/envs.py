import gym
from gym.utils import seeding
import numpy as np
import torch

class EconModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.beta = cfg.BETA
        self.gamma = cfg.GAMMA

    def u(self, c):
        return c**(1 - self.gamma) / (1 - self.gamma)

    def u_prime(self, c):
        return c**(-self.gamma)

    def v_star(self, x):
        return (1 - self.beta**(1 / self.gamma))**(-self.gamma) * self.u(x)

    def c_star(self, x):
        return (1 - self.beta**(1/self.gamma)) *  x


class CakeEatEnv(EconModel):
    
    def __init__(self, cfg):
        EconModel.__init__(self, cfg)

        self.observation_space = gym.spaces.Box(low=np.array([cfg.ENV_STATE_SPACE_L], dtype=np.float32),
                                high=np.array([cfg.ENV_STATE_SPACE_H], dtype=np.float32))
        self.action_space = gym.spaces.Box(low=np.array([cfg.ENV_ACTION_SPACE_L], dtype=np.float32), 
                              high=np.array([cfg.ENV_ACTION_SPACE_H], dtype=np.float32))

        # seeding
        self.np_random, _ = seeding.np_random(cfg.seed)
        self.observation_space.seed(cfg.seed)
        self.action_space.seed(cfg.seed)

    def gen_states(self, sampling=None, B=None):

        if sampling is None:
            sampling = self.cfg.ENV_GEN_STATE_SAMPLING
        if B is None:
            B = self.cfg.BATCH_SIZE
        # get state lower and higher bound for sampling
        l = torch.tensor(self.observation_space.low)
        u = torch.tensor(self.observation_space.high)
        # do the sampling
        if sampling == True:
            STATE_BATCH = torch.rand((B, 1)) * (u - l) + l
        else:
            STATE_BATCH = torch.linspace(l.item(), u.item(), B).unsqueeze(-1)

        return STATE_BATCH

    def transit(self, s, a, num_steps=None):

        if num_steps is None:
            num_steps = self.cfg.ENV_TRANSITION_NUM_STEPS
        R = 0
        beta = self.cfg.BETA
        for i in range(num_steps): 
            c = a
            r = self.u(c)
            s = s - c
            R += r*beta**i

        return R, s