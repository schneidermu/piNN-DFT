import torch
from torch import nn

device = torch.device('cpu')
true_constants_PBE = torch.Tensor([[0.06672455,
                                    (1 - torch.log(torch.Tensor([2])))/(torch.pi**2),
                                    1.709921,
                                    7.5957, 14.1189, 10.357,
                                    3.5876, 6.1977, 3.6231,
                                    1.6382, 3.3662,  0.88026,
                                    0.49294, 0.62517, 0.49671,
                                    # 1,  1,  1,
                                    0.031091, 0.015545, 0.016887,
                                    0.21370,  0.20548,  0.11125,
                                    -3/8*(3/torch.pi)**(1/3)*4**(2/3),
                                    0.8040,
                                    0.2195149727645171]]).to(device)

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
            nn.Dropout(p=dropout))
        
    def forward(self, x):
        residue = x

        return self.fc(self.fc(x)) + residue # skip connection 


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, dropout, DFT=None):
        super().__init__()

        self.DFT = DFT
        
        modules = []
        modules.extend([nn.Linear(7, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.0)])
        
        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim, dropout))
            
        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)
    
    def custom_sigmoid(self, x):
        # Custom sigmoid translates from [-inf, +inf] to [0, 4]
        # from 0 to 1
        result = (1+torch.e+(torch.e-3)/3)/(1 + (torch.e-3)/3 + torch.e**(-0.5*x+1))
        return result

    def forward(self, x):
        x = self.hidden_layers(x)

        
        if self.DFT == 'SVWN': # constraint for VWN3's Q_vwn function to 4*c - b**2 > 0
            constants = []
            for b_ind, c_ind in zip((2, 3, 11, 12, 13),(4, 5, 14, 15, 16)):
                 constants.append(torch.abs(x[:, c_ind]) + (x[:, b_ind]**2)/4 + 1e-5)
            x = torch.cat([x[:,0:4], torch.stack(constants[0:2], dim=1), x[:,6:14], torch.stack(constants[2:], dim=1), x[:,17:]], dim=1)
            del constants
        if self.DFT == 'PBE':
            ''' Use sigmoid on predictions and multiply by known constants for easier predictions '''
            x = self.custom_sigmoid(x)
            x = x * true_constants_PBE
        if self.DFT == 'XALPHA':
            x = x * 1.05
        return x


class pcPBEMLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants_x=2, nconstants_c=21, dropout=0.4, DFT=None):
        super().__init__()

        self.DFT = DFT

        modules_x = []
        modules_c = []
        modules_x.extend(
            [
                nn.Linear(5, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.0),
            ]
        )

        modules_c.extend(
            [
                nn.Linear(7, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.0),
            ]
        )

        for _ in range(num_layers // 2 - 1):
            modules_x.append(ResBlock(h_dim, dropout))
            modules_c.append(ResBlock(h_dim, dropout))

        modules_x.append(nn.Linear(h_dim, nconstants_x, bias=True))
        modules_c.append(nn.Linear(h_dim, nconstants_c, bias=True))

        self.hidden_layers_x = nn.Sequential(*modules_x)
        self.hidden_layers_c = nn.Sequential(*modules_c)

    
    def nonzero_custom_sigmoid(self, x):
        a = 47/300
        exp = torch.e**(0.5*x)
        # Custom sigmoid translates from [-inf, +inf] to [0.05, 4]
        result = (4*exp+a)/(3+a+exp)
        return result
    
    def kappa_activation(self, x):
        # Translates from [-inf, +inf] to [0, 1]
        return hardtanh(x+1)

    def kappa_sigmoid(self, x):
        a = 0.0526
        exp = torch.e**(0.5*x)
        # Custom sigmoid translates from [-inf, +inf] to [0.05, 1]
        result = (exp+a)/(1+a+exp)
        return result

    
    def infinite_activation(self, x):
        log2 = torch.log(torch.Tensor([2]))
        exponent = 1 + torch.e**(2*x*log2)

        return 1/log2*(torch.log(1+exponent))
    
    # 0,1 - rho alpha beta
    # 2,3,4 sigma aa ab bb
    # 5,6 tau a tau b
    def get_exchange_constants(self, x):

        x_x = x[:, 2:]

        x_x = self.hidden_layers_x(x_x)

        mu = x_x[:, 1]
        kappa = self.kappa_activation(x_x[:, 0])

        return mu, kappa


    def get_correlation_constants(self, x):

        x_c = self.hidden_layers_c(x)

        beta = x_c[:, 0] 
        gamma = x_c[:, 1]
        lda_c_params = x_c[:, 2:]

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
        return torch.hstack([torch.ones([x.shape[0], 2]).to(device), x[:, 2:]])


    def forward(self, x):
        mu, kappa = self.get_exchange_constants(x)
        beta, gamma, lda_c_params = self.get_correlation_constants(x)

        beta = self.nonzero_custom_sigmoid((beta - self.get_correlation_constants(self.all_sigma_zero(x))[0]).view(-1,1))
        gamma = self.nonzero_custom_sigmoid((gamma - self.get_correlation_constants(self.all_rho_inf(x))[1]).view(-1,1))
        lda_c_params = self.nonzero_custom_sigmoid(lda_c_params-self.get_correlation_constants(self.all_sigma_inf(x))[2])

        mu = self.nonzero_custom_sigmoid((mu - self.get_exchange_constants(self.all_sigma_zero(x))[0])).view(-1,1)

        kappa = kappa.view(-1,1)

        c_arr = torch.hstack([beta, gamma, lda_c_params, torch.ones([x.shape[0], 1]).to(device), kappa, mu])*true_constants_PBE

        return c_arr



def NN_XALPHA_model(num_layers=32, h_dim=32, nconstants=1, dropout=0.4, DFT='XALPHA'):
    return MLOptimizer(num_layers, h_dim, nconstants, dropout, DFT)

def NN_PBE_model(num_layers=8, h_dim=32, dropout=0.4, DFT='PBE'):
    return pcPBEMLOptimizer(num_layers, h_dim, dropout, DFT)
