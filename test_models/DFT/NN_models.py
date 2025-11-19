"""
Neural network models for test_models - re-exported from train_models.

This module re-exports the NN models from train_models for use in test_models.
Both packages use identical base implementations. If test_models needs different
behavior, specific classes and methods can be overridden here.
"""

import sys
from pathlib import Path

train_models_path = Path(__file__).parent.parent.parent / "train_models"
sys.path.insert(0, str(train_models_path))

from NN_models import MLOptimizer
from NN_models import pcPBEdoublestar
from NN_models import pcPBEstar
from NN_models import pcPBELMLOptimizer
from NN_models import pcPBEMLOptimizer

def NN_XALPHA_model(num_layers=6, h_dim=128, nconstants=1, dropout=0.0, DFT="XALPHA"):
    return MLOptimizer(
        num_layers=num_layers,
        h_dim=h_dim,
        nconstants=nconstants,
        dropout=dropout,
        DFT=DFT,
    )


def NN_PBE_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEMLOptimizer(
        num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT
    )


def NN_PBE_L_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBELMLOptimizer(
        num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT
    )


def NN_PBE_star_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEstar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT)


def NN_PBE_star_star_model(num_layers=6, h_dim=32, dropout=0.0, DFT="PBE"):
    return pcPBEdoublestar(num_layers=num_layers, h_dim=h_dim, dropout=dropout, DFT=DFT)
