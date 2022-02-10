import torch
import torch.nn as nn
from utils import soft_update, hard_update
from copy import deepcopy


class Agent:

    def __init__(self, env, actor, critic, actor_optim, critic_optim):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.discount = env.beta

    def update_critic(self, data, lr):
        """     
        update critic params once by Bellman equation Q(s, a) = r + b^n * Q(s', a')
        inputs are (s, a, r, s', a', n) and lr
        """

        # unpack the input
        s, a, r, next_s, next_a, n = data
        self.critic_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.critic_optim.zero_grad()
        
        # get the Q values and the Bellman error
        Q = self.critic(s, a)
        with torch.no_grad():
            next_Q = self.critic(next_s, next_a)
        target = r + self.discount**n * next_Q

        # do the update and return the loss
        q_loss = nn.MSELoss()(Q, target)
        q_loss.backward()
        self.critic_optim.step()
        return q_loss.item()

    def update_actor(self, data, lr):
        """     
        update actor params once by increasing the Q(s, a)
        inputs are d and lr
        """

        # unpack the input
        s = data
        a = self.actor(s)
        self.actor_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.actor_optim.zero_grad()
        
        # get the Q value, do the update and return the loss
        Q = self.critic(s, a)
        policy_loss = -Q.mean()
        policy_loss.backward()
        self.actor_optim.step()
        return policy_loss.item()

    # compare the difference of Q w.r.t. optimal strategy
    def test_critic(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.v_star(test_states)
        with torch.no_grad():
            Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        return test_q_loss.item()

    # compare the difference of strategy
    def test_actor(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.c_star(test_states)
        with torch.no_grad():
            a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        return test_policy_loss.item()

    # check whether given the optimal strategy (c*), 
    # the critic can well approximate the v* by supervised learning
    def train_critic_by_sl(self, lr):
        self.critic_optim.zero_grad()
        self.critic_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.v_star(test_states)
        Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        test_q_loss.backward()
        self.critic_optim.step()
        return test_q_loss.item()

    # check whether the actor can well approximate the c* by supervised learning
    def train_actor_by_sl(self, lr):

        self.actor_optim.zero_grad()
        self.actor_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.c_star(test_states)
        a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        test_policy_loss.backward()
        self.actor_optim.step()
        return test_policy_loss.item()


# this is the agent with target actor and target critic

class TargetAgent:

    def __init__(self, env, actor, critic, actor_optim, critic_optim):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.tgt_actor = deepcopy(actor)
        self.tgt_critic = deepcopy(critic)
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.discount = env.beta

    def update_critic(self, data, lr):
        """     
        update critic params once by Bellman equation Q(s, a) = r + b^n * Q(s', a')
        inputs are (s, a, r, s', a', n) and lr
        """

        # unpack the input
        s, a, r, next_s, next_a, n = data
        self.critic_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.critic_optim.zero_grad()
        
        # get the Q values and the Bellman error
        Q = self.critic(s, a)
        with torch.no_grad():
            next_Q = self.tgt_critic(next_s, next_a)
        target = r + self.discount**n * next_Q

        # do the update and return the loss
        q_loss = nn.MSELoss()(Q, target)
        q_loss.backward()
        soft_update(self.tgt_critic, self.critic, tau=0.01)
        self.critic_optim.step()

        return q_loss.item()

    def update_actor(self, data, lr):
        """     
        update actor params once by increasing the Q(s, a)
        inputs are d and lr
        """

        # unpack the input
        s = data
        a = self.actor(s)
        self.actor_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.actor_optim.zero_grad()
        
        # get the Q value, do the update and return the loss
        Q = self.critic(s, a)
        policy_loss = -Q.mean()
        policy_loss.backward()
        self.actor_optim.step()
        soft_update(self.tgt_actor, self.actor, tau=0.01)
        return policy_loss.item()

    # compare the difference of Q w.r.t. optimal strategy
    def test_critic(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.v_star(test_states)
        with torch.no_grad():
            Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        return test_q_loss.item()

    # compare the difference of strategy
    def test_actor(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.c_star(test_states)
        with torch.no_grad():
            a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        return test_policy_loss.item()

    # check whether given the optimal strategy (c*), 
    # the critic can well approximate the v* by supervised learning
    def train_critic_by_sl(self, lr):
        self.critic_optim.zero_grad()
        self.critic_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.v_star(test_states)
        Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        test_q_loss.backward()
        self.critic_optim.step()
        return test_q_loss.item()

    # check whether the actor can well approximate the c* by supervised learning
    def train_actor_by_sl(self, lr):

        self.actor_optim.zero_grad()
        self.actor_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.c_star(test_states)
        a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        test_policy_loss.backward()
        self.actor_optim.step()
        return test_policy_loss.item()



# this is the agent with adaptive sampling

class SamplingAgent:

    def __init__(self, env, actor, critic, actor_optim, critic_optim):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.tgt_actor = deepcopy(actor)
        self.tgt_critic = deepcopy(critic)
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.discount = env.beta

    def update_critic(self, data, lr, importance):
        """     
        update critic params once by Bellman equation Q(s, a) = r + b^n * Q(s', a')
        inputs are (s, a, r, s', a', n) and lr
        """

        # unpack the input
        s, a, r, next_s, next_a, n = data
        self.critic_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.critic_optim.zero_grad()
        
        # get the Q values and the Bellman error
        Q = self.critic(s, a)
        with torch.no_grad():
            next_Q = self.tgt_critic(next_s, next_a)
        target = r + self.discount**n * next_Q

        # do the update and return the loss
        q_losses = (Q - target) ** 2
        mean_q_loss = torch.sum(q_losses * importance)
        mean_q_loss.backward()
        soft_update(self.tgt_critic, self.critic, tau=0.01)
        self.critic_optim.step()

        return mean_q_loss.item(), q_losses

    def update_actor(self, data, lr):
        """     
        update actor params once by increasing the Q(s, a)
        inputs are d and lr
        """

        # unpack the input
        s = data
        a = self.actor(s)
        self.actor_optim.param_groups[0]['lr'] = lr
        
        # clear the gradients
        self.actor_optim.zero_grad()
        
        # get the Q value, do the update and return the loss
        Q = self.critic(s, a)
        policy_loss = -Q.mean()
        policy_loss.backward()
        self.actor_optim.step()
        soft_update(self.tgt_actor, self.actor, tau=0.01)
        return policy_loss.item()

    # compare the difference of Q w.r.t. optimal strategy
    def test_critic(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.v_star(test_states)
        with torch.no_grad():
            Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        return test_q_loss.item()

    # compare the difference of strategy
    def test_actor(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.c_star(test_states)
        with torch.no_grad():
            a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        return test_policy_loss.item()

    # check whether given the optimal strategy (c*), 
    # the critic can well approximate the v* by supervised learning
    def train_critic_by_sl(self, lr):
        self.critic_optim.zero_grad()
        self.critic_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.v_star(test_states)
        Q = self.critic(test_states, self.env.c_star(test_states))
        test_q_loss = nn.MSELoss()(target, Q)
        test_q_loss.backward()
        self.critic_optim.step()
        return test_q_loss.item()

    # check whether the actor can well approximate the c* by supervised learning
    def train_actor_by_sl(self, lr):

        self.actor_optim.zero_grad()
        self.actor_optim.param_groups[0]['lr'] = lr

        test_states = self.env.gen_states()
        target = self.env.c_star(test_states)
        a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        test_policy_loss.backward()
        self.actor_optim.step()
        return test_policy_loss.item()