import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from models import QNetwork, PolicyNetwork
from utils import configObj, LR_SCHEME
from envs import CakeEatEnv

config_name = '/Users/yaolangzhong/PycharmProjects/RLEcon/CakeEat/DDPG/config'
cfg = configObj(config_name,verbose=False)

env = CakeEatEnv(cfg)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# build the agent
critic = QNetwork(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.shape[0],
                  hidden_dim=cfg.CRITIC_HIDDEN_DIM,
                  num_hidden_layer=cfg.CRITIC_NUM_HIDDEN_LAYER,
                  activate=cfg.CRITIC_ACTIVATE,
                  batch_norm=cfg.CRITIC_BATCH_NORM,
                  weight_init=cfg.CRITIC_WEIGHT_INIT)

actor = PolicyNetwork(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.shape[0],
                  hidden_dim=cfg.ACTOR_HIDDEN_DIM,
                  num_hidden_layer=cfg.ACTOR_NUM_HIDDEN_LAYER,
                  activate=cfg.ACTOR_ACTIVATE,
                  batch_norm=cfg.ACTOR_BATCH_NORM,
                  weight_init=cfg.ACTOR_WEIGHT_INIT)

critic_optim = optim.Adam(critic.parameters(), lr=cfg.MAX_CRITIC_LR)
actor_optim = optim.Adam(actor.parameters(), lr=cfg.MAX_ACTOR_LR)

agent = Agent(env=env, critic=critic, actor=actor, critic_optim=critic_optim, actor_optim=actor_optim)


lr_scheme = LR_SCHEME(cfg)


if __name__ == '__main__':

    for i in range(cfg.MAX_ITERATION):
        s = env.gen_states()
        a = agent.actor(s)
        r, s_ = env.transit(s, a)
        a_ = agent.actor(s_)
        data = [s, a, r, s_, a_, cfg.ENV_TRANSITION_NUM_STEPS]
        train_q_loss= agent.update_critic(data, lr_scheme(i)[0])
        if i % cfg.REPORT_PER_ITERATION == 0:
            print(f"Epoch {i}: lr {lr_scheme(i)[0]:.4f}, train loss {train_q_loss:.6f}\n")

