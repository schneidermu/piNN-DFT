"""F016 diagnostic-only bounded q representations; never a production model."""
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train_models'))
import torch
from NN_models_lap import pcPBELMLOptimizerV2Lap
from torch import nn

from dft_functionals.constants import EPS_RHO
from train_models.lap_vxc import LapEnergy


class DiagnosticEnergy(LapEnergy):
    """Reuse exact LapEnergy forward/operator dispatch; separate clone protocol."""
    def __init__(self, model):
        nn.Module.__init__(self)
        if not isinstance(model, DiagnosticQModel):
            raise TypeError('F016 diagnostic clone required')
        self.model = model


def raw_q(x):
    return x[:, 7:9] / (4 * (3 * math.pi**2)**(2/3) * (x[:, :2] + EPS_RHO)**(5/3))


def descriptors(x, representation):
    base = pcPBELMLOptimizerV2Lap.get_correlation_descriptors(x)
    if representation == 'R0':
        return base
    wide = torch.tanh(raw_q(x) / 4)
    if representation == 'R1':
        return torch.cat([base[:, :5], wide, base[:, 7:8]], 1)
    if representation == 'R2':
        return torch.cat([base, wide], 1)
    raise ValueError(representation)


class DiagnosticQModel(pcPBELMLOptimizerV2Lap):
    """R2 preserves the ORIGINAL metric in Gc's distance-weighted correction.

    Merely zero-padding network weights while expanding that metric is NOT an
    exact embedding. Extra features participate in network inputs, not metric.
    Version identifier is separate; no checkpoint of this clone is written.
    """
    def __init__(self, source, representation):
        super().__init__(**source.model_kwargs)
        self.to(next(source.parameters()))
        self.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
        if representation not in ('R1', 'R2'):
            raise ValueError(representation)
        self.representation = representation
        self.architecture = f'F016-diagnostic-{representation}-original-anchor-metric-v1'
        self.descriptor_protocol = f'F016-{representation}-q-scale4-v1'
        if representation == 'R2':
            for stack, width in ((self.c_input_layers, 10), (self.x_feature_extractor, 3)):
                old = stack[0]
                new = nn.Linear(width, old.out_features, bias=False).to(old.weight)
                with torch.no_grad():
                    new.weight.zero_()
                    new.weight[:, :old.in_features].copy_(old.weight)
                stack[0] = new

    def get_correlation_descriptors(self, x):
        return descriptors(x, self.representation)

    get_exchange_descriptors = get_correlation_descriptors

    @staticmethod
    def all_sigma_zero(x):
        out = pcPBELMLOptimizerV2Lap.all_sigma_zero(x[:, :8])
        return torch.cat([out, torch.zeros_like(x[:, 8:])], 1)

    @staticmethod
    def all_rho_inf(x):
        return torch.cat([torch.ones_like(x[:, :2]), x[:, 2:]], 1)

    @staticmethod
    def all_s_inf(x):
        return torch.cat([x[:, :2], torch.ones_like(x[:, 2:5]), x[:, 5:]], 1)

    @staticmethod
    def swap_descriptors(d):
        original = pcPBELMLOptimizerV2Lap.swap_descriptors(d[:, :8])
        return torch.cat([original, d[:, [9, 8]]], 1) if d.shape[1] == 10 else original

    def exchange_hidden(self, d):
        if self.representation == 'R1':
            return super().exchange_hidden(d)
        # Evaluate the old contraction plus separate new contribution. Zero
        # weights therefore preserve the original contraction's reduction order.
        first = self.x_feature_extractor[0]
        def branch(old, extra):
            h = nn.functional.linear(d[:, old], first.weight[:, :2])
            h = h + nn.functional.linear(d[:, extra], first.weight[:, 2:])
            for layer in list(self.x_feature_extractor)[1:]:
                h = layer(h)
            return h
        return branch([2, 5], [8]), branch([4, 6], [9])

    def correlation_output(self, d, hx):
        if self.representation == 'R1':
            return super().correlation_output(d, hx)
        first = self.c_input_layers[0]
        def embedded(x):
            h = nn.functional.linear(x[:, :8], first.weight[:, :8])
            h = h + nn.functional.linear(x[:, 8:], first.weight[:, 8:])
            for layer in list(self.c_input_layers)[1:]:
                h = layer(h)
            return self.c_symmetrization_blocks(h)
        h = self.c_post_symm_blocks((embedded(d) + embedded(self.swap_descriptors(d))) / 2)
        return self.c_output_layer(torch.cat([h, hx], 1))

    @staticmethod
    def _lagrange_correct_Gc(real, c1, c2, c3, d, x1, x2, x3, delta=1.):
        return pcPBELMLOptimizerV2Lap._lagrange_correct_Gc(
            real, c1, c2, c3, d[:, :8], x1[:, :8], x2[:, :8], x3[:, :8], delta)
