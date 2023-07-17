from common import *

class DeepQNetwork(nn.Module):
    """ Building a Deep Q Network with Pytorch """
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ asa """
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """ """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        

class DuelingQNetwork(nn.Module):
    """ Building a Deep Q Network with Pytorch """
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ asa """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3_a = nn.Linear(fc2_units, action_size)
        self.fc3_v = nn.Linear(fc2_units, 1)
        
    def forward(self, state):
        """ """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        v = F.relu(self.fc3_v(x))
        a = F.relu(self.fc3_a(x))

        q = v + a - a.mean()
        return q
        