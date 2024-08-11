import torch
from torch import nn
import random
import numpy as np
import gc

random.seed(42)

device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
true_constants_PBE = torch.Tensor(
    [
        [
            0.06672455,
            (1 - torch.log(torch.Tensor([2]))) / (torch.pi**2),
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
            -3 / 8 * (3 / torch.pi) ** (1 / 3) * 4 ** (2 / 3),
            0.8040,
            0.2195149727645171,
        ]
    ]
).to(device)
hardtanh = torch.nn.Hardtanh(min_val=0.05, max_val=1)

"""
Define an nn.Module class for a simple residual block with equal dimensions
"""


class ResBlock(nn.Module):

    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers
    """

    def __init__(self, h_dim, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        residue = x

        return self.fc(self.fc(x)) + residue  # skip connection

      
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
                nn.Dropout(p=0.0),
            ]
        )

        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim, dropout))

        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)


    def dm21_like_sigmoid(self, x):
        exp = torch.e**(0.5*x)
        # Custom sigmoid translates from [-inf, +inf] to [0, 2] as in DM21 paper
        return 2*exp/(1+exp)


    def forward(self, x):
        if self.training:
            x = random.choice([x, x[:, [1, 0, 4, 3, 2, 6, 5]]]) # Randomly flip spin
        x = self.hidden_layers(x)
        x = 1.05*self.dm21_like_sigmoid(x)

        return x


class pcPBEMLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants_x=2, nconstants_c=21, dropout=0.4, DFT=None):
        super().__init__()

        self.DFT = DFT

        modules_x = [] # NN part for exchange
        modules_c = [] # NN part for correlation

        input_layer_c = [
                nn.Linear(7, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.0),
            ]
        
        input_layer_x = [
                nn.Linear(5, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.0),
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

        # Translates from [-inf, +inf] to [0, 1]
        return hardtanh(x+1)

    # 0,1 rho alpha beta
    # 2,3 s_alpha, s_beta
    # 4,5 tau a tau b

    def get_exchange_constants(self, x):

        x_x = self.hidden_layers_x(x[:, 2:])

        mu = x_x[:, 1]
        kappa = self.kappa_activation(x_x[:, 0])

        del x_x
        gc.collect()
        torch.cuda.empty_cache()

        return mu, kappa


    def get_correlation_constants(self, x):

        x_c = self.hidden_layers_c(x)

        beta = x_c[:, 0] 
        gamma = x_c[:, 1]
        lda_c_params = x_c[:, 2:]

        del x_c
        gc.collect()
        torch.cuda.empty_cache()
        
        return beta, gamma, lda_c_params

    
    @staticmethod
    def all_sigma_zero(x):
        # For betta
        return torch.hstack([x[:, :2], torch.zeros([x.shape[0], 5]).to(device)])
    
    @staticmethod
    def all_sigma_inf(x):
        # For lda_c
        return torch.hstack([x[:, :2], torch.ones([x.shape[0], 5]).to(device)])
    
    @staticmethod
    def all_rho_inf(x):
        # For gamma
        # Then rs is zero
        return torch.hstack([torch.zeros([x.shape[0], 2]).to(device), x[:, 2:]])

    @staticmethod
    def custom_relu(x):
        return torch.nn.functional.relu(x+0.95)+0.05

    def forward(self, x):
        if self.training:
            x = random.choice([x, x[:, [1, 0, 4, 3, 2, 6, 5]]])

        mu, kappa = self.get_exchange_constants(x)
        beta, gamma, lda_c_params = self.get_correlation_constants(x)

        beta = self.custom_relu((beta - self.get_correlation_constants(self.all_sigma_zero(x))[0]).view(-1,1))
        gamma = self.custom_relu((gamma - self.get_correlation_constants(self.all_rho_inf(x))[1]).view(-1,1))
        lda_c_params = self.custom_relu(lda_c_params-self.get_correlation_constants(self.all_sigma_inf(x))[2])

        mu = self.custom_relu((mu - self.get_exchange_constants(self.all_sigma_zero(x))[0])).view(-1,1)

        kappa = kappa.view(-1,1)

        c_arr = torch.hstack([beta, gamma, lda_c_params, torch.ones([x.shape[0], 1]).to(device), kappa, mu])*true_constants_PBE

        del mu, beta, gamma, kappa, lda_c_params
        gc.collect()
        torch.cuda.empty_cache()

        return c_arr