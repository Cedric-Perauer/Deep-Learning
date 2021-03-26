import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

"""
this is the architecutre from the DQN paper
class QNetwork(nn.Module):
    #Actor (Policy) Model.

    def __init__(self, state_size, action_size, seed):
        #Initialize parameters and build model.
        #Params
        #======
        #    state_size (int): Dimension of each state
        #    action_size (int): Dimension of each action
        #    seed (int): Random seed
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=state_size,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3) 
        self.fc1 = nn.Linear(3136,512)    
        self.fc2 = nn.Linear(512,action_size) 
                                          
        self.relu = nn.ReLU()             
     
                                          
    def forward(self, state):   
        print(state.shape)
        #Build a network that maps state -> action values.
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x)) 
        x = self.relu(self.conv3(x))  
        x = torch.view(-1,3136)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
"""