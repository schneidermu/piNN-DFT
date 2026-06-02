"""
Compatibility adapter for inference-time NN model constructors.

The training code has moved to a newer model API, but test_models still imports
named factory functions from this module. Keep those imports stable here and
map them to the currently available training architectures.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


TRAIN_MODELS_PATH = Path(__file__).parent.parent.parent / "train_models" / "NN_models.py"
SPEC = spec_from_file_location("train_models_nn_models_runtime", TRAIN_MODELS_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load training model definitions from {TRAIN_MODELS_PATH}")
TRAIN_MODELS = module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_MODELS)


def _unsupported(name: str):
    raise NotImplementedError(
        f"{name} is not available with the current train_models/NN_models.py API. "
        "Only the NN_PBE-L-compatible optimizer is currently exposed."
    )


def NN_XALPHA_model(*args, **kwargs):
    return _unsupported("NN_XALPHA_model")


def NN_PBE_model(*args, **kwargs):
    return _unsupported("NN_PBE_model")


def NN_PBE_L_model(
    num_layers=6,
    h_dim=32,
    dropout=0.0,
    DFT="PBE",
    gc_svelu_mirror=False,
    **kwargs,
):
    model_class = (
        TRAIN_MODELS.pcPBELMLOptimizerV2GcSveluMirror
        if gc_svelu_mirror
        else TRAIN_MODELS.pcPBELMLOptimizerV2
    )
    return model_class(
        num_layers=num_layers,
        h_dim=h_dim,
        dropout=dropout,
        DFT=DFT,
        **kwargs,
    )


def NN_PBE_star_model(*args, **kwargs):
    return _unsupported("NN_PBE_star_model")


def NN_PBE_star_star_model(*args, **kwargs):
    return _unsupported("NN_PBE_star_star_model")
