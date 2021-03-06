import torch
import torch.nn as nn


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
        return test_q_loss

    # compare the difference of strategy
    def test_actor(self):
        test_states = self.env.gen_states(sampling=False, B=100)
        target = self.env.c_star(test_states)
        with torch.no_grad():
            a = self.actor(test_states)
        test_policy_loss = nn.MSELoss()(target, a)
        return test_policy_loss




