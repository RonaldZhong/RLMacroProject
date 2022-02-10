import yaml
import numpy as np
import pandas as pd
import copy

def yaml_parser(config_name):
    with open(config_name + '.yaml', 'r') as f:
        config_content = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config_content

class config_settings:
    def set_var(self, var, verbose):
        if verbose == True:
            print('\nConfig settings:')
            print('-----------------')
        for key, value in var.items():
            setattr(self, key, value)
            if verbose == True:
                print(f'{key:<20}: {str(value):<20}')

def configObj(config_name, verbose=False):
    config_set = config_settings()
    config_set.set_var(yaml_parser(config_name), verbose)
    return config_set


class LR_SCHEME:
    def __init__(self, cfg):
        self.min_crt_lr = cfg.MIN_CRITIC_LR
        self.max_crt_lr = cfg.MAX_CRITIC_LR
        self.crt_decay = cfg.CRITIC_LR_DECAY

        self.min_act_lr = cfg.MIN_ACTOR_LR
        self.max_act_lr = cfg.MAX_ACTOR_LR
        self.act_decay = cfg.ACTOR_LR_DECAY
    def __call__(self, i):
        crt_lr = self.min_crt_lr + (self.max_crt_lr - self.min_crt_lr) * np.exp(-self.crt_decay*i)
        act_lr = self.min_act_lr + (self.max_act_lr - self.min_act_lr) * np.exp(-self.act_decay*i)
        return crt_lr, act_lr


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)