
# this test train agent's critic and actor both by naive supervised learning

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dpsolver import DPSolver
from agent import Agent
from utils import configObj
from models import QNetwork, PolicyNetwork
from envs import CakeEatEnv
from utils import LR_SCHEME


config_name = 'config'
cfg = configObj(config_name,verbose=False)
env = CakeEatEnv(cfg)
OPTIMAL_ACTION = env.c_star(1)
lr_scheme = LR_SCHEME(cfg)


dp = DPSolver(cfg)
dp.solve(env.c_star)

def plot(ax, model, label):
    s = torch.linspace(cfg.ENV_STATE_SPACE_L, cfg.ENV_STATE_SPACE_H, 100).unsqueeze(-1)
    a = env.c_star
    with torch.no_grad():
        v = model(s, a(s))
    s = s.squeeze(-1).detach().numpy()
    v = v.squeeze(-1).detach().numpy()
    ax.plot(s, v, label=label);
    ax.legend(fontsize=20);
    

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

# train critic
for i in range(1000):
    train_q_loss= agent.train_critic_by_sl(lr_scheme(i)[0])
    test_q_loss = agent.test_critic()
    if i % cfg.REPORT_PER_ITERATION == 0:
        print(f"Epoch {i}:\n lr {lr_scheme(i)[0]:.4f}, train q loss {train_q_loss:.4f}\n\
        test q loss {test_q_loss:.4f}\n\n")
        
# test performance by plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plot(ax, dp, label='Dynamic Programming')
plot(ax, agent.critic, label='Neural Network')
s = np.linspace(cfg.ENV_STATE_SPACE_L, cfg.ENV_STATE_SPACE_H, 100)
ax.plot(s, env.v_star(s), label='True V')
ax.legend(fontsize=20);

# train actor
for i in range(10000):
    train_policy_loss= agent.train_actor_by_sl(lr_scheme(i)[0])
    test_policy_loss = agent.test_actor()
    if i % cfg.REPORT_PER_ITERATION == 0:
        print(f"Epoch {i}:\n lr {lr_scheme(i)[0]:.4f}, train policy loss {train_policy_loss:.6f}\n\
        test policy loss {test_policy_loss:.6f}\n\n")

# test performance by plotting
s = torch.linspace(cfg.ENV_STATE_SPACE_L, cfg.ENV_STATE_SPACE_H, 100).unsqueeze(-1)
true_a = env.c_star(s)
a = agent.actor(s)

s = s.squeeze(-1).detach().numpy()
true_a = true_a.squeeze(-1).detach().numpy()
a = a.squeeze(-1).detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(s, true_a, label='Optimal Policy')
ax.plot(s, a, label='Actor Approximated Policy')
ax.legend(fontsize=20);
