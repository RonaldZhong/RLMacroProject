import torch
import torch.nn as nn

# Initialize weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=4, num_hidden_layer=1, activate='relu', 
    batch_norm=True, weight_init=False):

        super(QNetwork, self).__init__()
        
        if activate=='relu':
            self.activate = nn.ReLU()
        if activate=='tanh':
            self.activate = nn.Tanh()
        if activate == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.1)

        self.hidden = nn.ModuleList()

        for i in range(num_hidden_layer):
                if batch_norm==True:
                    self.hidden.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
                self.hidden.append(nn.Linear(hidden_dim, hidden_dim)),
                self.hidden.append(self.activate)
        if batch_norm==True:
                self.hidden.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))

        self.net = nn.Sequential(               
            nn.Linear(state_dim+action_dim, hidden_dim), 
            self.activate, 
            *self.hidden,
            nn.Linear(hidden_dim, 1)
        )  
        self.net.apply(weights_init_)

    def forward(self, state, action):

        x = torch.cat([state, action], 1)

        return self.net(x)



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=4, num_hidden_layer=1, activate='relu', 
    batch_norm=True, weight_init=False):

        super(PolicyNetwork, self).__init__()


        if activate=='relu':
            self.activate = nn.ReLU()
        if activate=='tanh':
            self.activate = nn.Tanh()
        if activate == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.1)

        self.hidden = nn.ModuleList()

        for i in range(num_hidden_layer):
                if batch_norm==True:
                    self.hidden.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
                self.hidden.append(nn.Linear(hidden_dim, hidden_dim)),
                self.hidden.append(self.activate)
        if batch_norm==True:
                self.hidden.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))

        self.net = nn.Sequential(               
            nn.Linear(state_dim, hidden_dim), 
            self.activate, 
            *self.hidden,
            nn.Linear(hidden_dim, action_dim)
        )  
        self.net.apply(weights_init_)

    def forward(self, x):

        return nn.Sigmoid()(self.net(x))