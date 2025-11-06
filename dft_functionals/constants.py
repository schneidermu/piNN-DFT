"""
Shared constants for DFT functionals and neural network models.

This module contains canonical definitions of constants used across
the piNN-DFT project
"""

import numpy as np
import torch

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
            0.8040,
            0.2195149727645171,
        ]
    ]
)


true_constants_SVWN3 = [
    0.0310907,
    0.01554535,
    3.72744,
    7.06042,
    12.9352,
    18.0578,
    -0.10498,
    -0.32500,
    0.0310907,
    0.01554535,
    -1 / (6 * np.pi**2),
    13.0720,
    20.1231,
    1.06835,
    42.7198,
    101.578,
    11.4813,
    -0.409286,
    -0.743294,
    -0.228344,
    1,
]


EPS_SIGMA = 1e-30
EPS_RHO = 1e-10
EPS_LOG = 1e-5
MGGA_DESCRIPTOR_DIMENSIONALITY = 7
MGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY = 2
LLMGGA_DESCRIPTOR_DIMENSIONALITY = 9
LLMGGA_DESCRIPTOR_EXCHANGE_DIMENSIONALITY = 3
RHO_ALPHA_INDEX = 0
RHO_BETA_INDEX = 1
S_ALPHA_INDEX = 2
S_BETA_INDEX = 4
S_TOTAL_INDEX = 3
TAU_ALPHA_INDEX = 5
TAU_BETA_INDEX = 6
LAPL_ALPHA_INDEX = 7
LAPL_BETA_INDEX = 8
MGGA_SPIN_INVERTED_SLICE = [
    RHO_BETA_INDEX,
    RHO_ALPHA_INDEX,
    S_BETA_INDEX,
    S_TOTAL_INDEX,
    S_ALPHA_INDEX,
    TAU_BETA_INDEX,
    TAU_ALPHA_INDEX,
]
LLMGGA_SPIN_INVERTED_SLICE = [
    RHO_BETA_INDEX,
    RHO_ALPHA_INDEX,
    S_BETA_INDEX,
    S_TOTAL_INDEX,
    S_ALPHA_INDEX,
    TAU_BETA_INDEX,
    TAU_ALPHA_INDEX,
    LAPL_BETA_INDEX,
    LAPL_ALPHA_INDEX,
]
MGGA_SPIN_SCALING_MULTIPLIER = torch.tensor([2, 2, 4, 4, 4, 2, 2])
LLMGGA_SPIN_SCALING_MULTIPLIER = torch.tensor([2, 2, 4, 4, 4, 2, 2, 2, 2])
BETA_CORR_INDEX = 0
GAMMA_CORR_INDEX = 1
MU_EX_INDEX = 0
KAPPA_EX_INDEX = 1
