import torch
from torch import nn
import random
import numpy as np

random.seed(42)

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
#device = torch.device("cpu")

true_constants_PBE = torch.Tensor(
    [
        [
            0.06672455,
            (1 - torch.log(torch.Tensor([2]))) / (np.pi**2),
            1.709921,
            7.5957,
            14.1189,
            10.357,
            3.5876,
            6.1977,
            3.6231,
            1.6382,
            3.3662,
            0.88026,
            0.49294,
            0.62517,
            0.49671,
            # 1,  1,  1,
            0.031091,
            0.015545,
            0.016887,
            0.21370,
            0.20548,
            0.11125,
            -3 / 8 * (3 / np.pi) ** (1 / 3) * 4 ** (2 / 3),
            0.8040,
            0.2195149727645171,
        ]
    ]
).to(device)
hardtanh = torch.nn.Hardtanh(min_val=0.01, max_val=1)

"""
Define an nn.Module class for a simple residual block with equal dimensions
"""


class ResBlock(nn.Module):

    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers
    """

    def __init__(self, h_dim, dropout):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        residue = x

        out = self.fc1(x)
        out = self.fc2(out)
        out += residue

        return out 


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, dropout, DFT=None, constants=[]):
        super().__init__()

        self.DFT = DFT
        self.constants = constants
        if self.constants:
            nconstants = len(constants)

        modules = []
        modules.extend(
            [
                nn.Linear(7, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ]
        )

        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim, dropout))

        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)


    def dm21_like_sigmoid(self, x):
        '''
        Custom sigmoid translates from [-inf, +inf] to [0, 2]
        '''

        exp = torch.exp(-0.5*x)
        return 2/(1+exp)


    def unsymm_forward(self, x):

        x = self.hidden_layers(x)
        x = 1.05*self.dm21_like_sigmoid(x)

        return x

    def forward(self, x):
        '''
        Returns:
            spin-symmetrized enhancement factor for LDA exhange energy
        '''
        
        result = (self.unsymm_forward(x)+self.unsymm_forward(x[:, [1, 0, 4, 3, 2, 6, 5]]))/2

        return result


class pcPBEMLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants_x=2, nconstants_c=2, dropout=0.4, DFT=None):
        super().__init__()

        self.DFT = DFT

        modules_x = [] # NN part for exchange
        modules_c = [] # NN part for correlation

        input_layer_c = [
                nn.Linear(7, h_dim, bias=False),
                nn.LayerNorm(h_dim),
                nn.PReLU(),
            ]
        
        input_layer_x = [
                nn.Linear(5, h_dim, bias=False),
                nn.LayerNorm(h_dim),
                nn.PReLU(),
            ]

        modules_x.extend(
            input_layer_x
        )
        modules_c.extend(
            input_layer_c
        )

        for _ in range(num_layers // 2 - 1):
            modules_x.append(ResBlock(h_dim, dropout))
            modules_c.append(ResBlock(h_dim, dropout))

        modules_x.append(nn.Linear(h_dim, nconstants_x, bias=True))
        modules_c.append(nn.Linear(h_dim, nconstants_c, bias=True))

        self.hidden_layers_x = nn.Sequential(*modules_x)
        self.hidden_layers_c = nn.Sequential(*modules_c)

    
    def kappa_activation(self, x):
        '''
        Translates values from [-inf, +inf] to [0, 1]
        '''
        return hardtanh(x+1)


    def get_exchange_constants(self, x):

        x_x = self.hidden_layers_x(x[:, 2:]) # Slice out density descriptors

        return x_x[:, 1], self.kappa_activation(x_x[:, 0])


    def get_correlation_constants(self, x):

        x_c = self.hidden_layers_c(x)

        return x_c[:, 0], x_c[:, 1]

    
    @staticmethod
    def all_sigma_zero(x):
        '''
        Function for parameter beta and mu constraint
        '''
        return torch.hstack([x[:, :2], torch.zeros([x.shape[0], 5]).to(x.device)])
    
    @staticmethod
    def all_sigma_inf(x):
        '''
        Function for PW91 correlation perameters constraint
        '''
        return torch.hstack([x[:, :2], torch.ones([x.shape[0], 5]).to(x.device)])
    
    @staticmethod
    def all_rho_inf(x):
        '''
        Function for parameter gamma constraint
        '''
        return torch.hstack([torch.zeros([x.shape[0], 2]).to(x.device), x[:, 2:]])

    @staticmethod
    def custom_relu(x):
        return torch.nn.functional.relu(x+0.99)+0.01


    def forward(self, x):
        if self.training:
            x = random.choice([x, x[:, [1, 0, 4, 3, 2, 6, 5]]])

        mu, kappa = self.get_exchange_constants(x)

        beta, gamma = self.get_correlation_constants(x)

        beta = self.custom_relu((beta - self.get_correlation_constants(self.all_sigma_zero(x))[0]).view(-1,1))
        gamma = self.custom_relu((gamma - self.get_correlation_constants(self.all_rho_inf(x))[1]).view(-1,1))
        mu = self.custom_relu((mu - self.get_exchange_constants(self.all_sigma_zero(x))[0])).view(-1,1)
        kappa = kappa.view(-1,1)

        return torch.hstack([beta, gamma, torch.ones([x.shape[0], 20]).to(x.device), kappa, mu])*true_constants_PBE.to(x.device)
