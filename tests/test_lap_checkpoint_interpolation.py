import torch
from torch import nn

from tools import lap_checkpoint_interpolation as interpolation


class Tied(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.up = nn.Linear(2, 2, bias=False)
        self.down = nn.Linear(2, 2, bias=False)
        self.down.weight = self.up.weight
        self.register_buffer('version', torch.tensor([1], dtype=torch.int64))
        with torch.no_grad():
            self.up.weight.copy_(left)
        self.register_buffer('scale', right)


def parents():
    left = Tied(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0.5]))
    right = Tied(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([0.5]))
    return left.state_dict(), right.state_dict()


def layout(module):
    names = []
    groups = {}
    order = []
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names.append(name)
        key = parameter.data_ptr()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(name)
    trainable = tuple(name for name, parameter in module.named_parameters(remove_duplicate=True) if parameter.requires_grad)
    buffers = tuple(name for name, _value in module.named_buffers())
    return trainable, tuple(tuple(groups[key]) for key in order), buffers


def test_interpolation_endpoints_and_quarter_points_are_exact():
    left, right = parents()
    module = Tied(torch.zeros((2, 2)), torch.tensor([0.5]))
    trainable, ties, buffers = layout(module)
    for alpha, expected in ((0.0, 1.0), (1.0, 5.0)):
        mixed = interpolation.interpolate_state(left, right, alpha, trainable, buffers, ties)
        source = left if alpha == 0.0 else right
        assert torch.equal(mixed['up.weight'], source['up.weight'])
        assert torch.equal(mixed['down.weight'], mixed['up.weight'])
    quarter = interpolation.interpolate_state(left, right, 0.25, trainable, buffers, ties)
    manual = ((1.0 - 0.25) * left['up.weight'].double() + 0.25 * right['up.weight'].double()).to(torch.float32)
    assert torch.equal(quarter['up.weight'], manual)
    half = interpolation.interpolate_state(left, right, 0.5, trainable, buffers, ties)
    assert torch.equal(half['up.weight'], ((left['up.weight'].double() + right['up.weight'].double()) / 2).to(torch.float32))
    three = interpolation.interpolate_state(left, right, 0.75, trainable, buffers, ties)
    assert three['up.weight'][0, 0].item() == torch.tensor(4.0).item()


def test_unknown_alpha_shape_buffer_and_nonfinite_are_rejected():
    left, right = parents()
    module = Tied(torch.zeros((2, 2)), torch.tensor([0.5]))
    trainable, ties, buffers = layout(module)
    for alpha in (-0.25, 0.2, 1.25):
        try:
            interpolation.interpolate_state(left, right, alpha, trainable, buffers, ties)
        except ValueError:
            pass
        else:
            raise AssertionError(alpha)
    broken = {name: value.clone() for name, value in right.items()}
    broken['up.weight'] = torch.zeros((3, 2))
    broken['down.weight'] = broken['up.weight']
    try:
        interpolation.interpolate_state(left, broken, 0.5, trainable, buffers, ties)
    except ValueError:
        pass
    else:
        raise AssertionError('shape')
    shifted = {name: value.clone() for name, value in right.items()}
    shifted['version'] = torch.tensor([2], dtype=torch.int64)
    try:
        interpolation.interpolate_state(left, shifted, 0.5, trainable, buffers, ties)
    except RuntimeError:
        pass
    else:
        raise AssertionError('buffer')
    poisoned = {name: value.clone() for name, value in left.items()}
    poisoned['up.weight'] = torch.tensor([[float('nan'), 2.0], [3.0, 4.0]])
    poisoned['down.weight'] = poisoned['up.weight']
    try:
        interpolation.interpolate_state(poisoned, right, 0.5, trainable, buffers, ties)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('nonfinite')


def test_parents_are_not_mutated_and_hashes_repeat():
    left, right = parents()
    module = Tied(torch.zeros((2, 2)), torch.tensor([0.5]))
    trainable, ties, buffers = layout(module)
    before = left['up.weight'].clone()
    first = interpolation.interpolate_state(left, right, 0.5, trainable, buffers, ties)
    second = interpolation.interpolate_state(left, right, 0.5, trainable, buffers, ties)
    assert torch.equal(left['up.weight'], before)
    assert interpolation.state_sha(first) == interpolation.state_sha(second)
    restored = Tied(torch.zeros((2, 2)), torch.tensor([0.5]))
    restored.load_state_dict(first)
    assert restored.up.weight.data_ptr() == restored.down.weight.data_ptr()


def test_eligibility_selection_and_incomplete_records():
    assert interpolation.eligibility({'relchem': 0.9, 'ae17': 0.2, 'exc': 0.3, 'op': 0.4})
    assert not interpolation.eligibility({'relchem': 1.0, 'ae17': 0.2, 'exc': 0.3, 'op': 0.4})
    rows = [
        {'alpha': 0.25, 'complete': True, 'eligible': False, 'clean28': 8.0},
        {'alpha': 0.5, 'complete': True, 'eligible': True, 'clean28': 9.0},
        {'alpha': 0.75, 'complete': True, 'eligible': True, 'clean28': 9.0},
    ]
    decision = interpolation.decide(rows)
    assert decision['classification'] == 'GO'
    assert decision['winner']['alpha'] == 0.5
    assert decision['best_clean28_candidate']['alpha'] == 0.25
    rows[0]['complete'] = False
    assert interpolation.decide(rows)['classification'] == 'PARTIAL'
    try:
        interpolation.require_complete({'status': 'partial', 'objectives': {}, 'reactions': []})
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete')


def test_driver_does_not_call_optimizer_or_backward():
    source = interpolation.Path(__file__).resolve().parents[1].joinpath(
        'tools/lap_checkpoint_interpolation.py'
    ).read_text(encoding='utf-8')
    interpolation.forbid_training_calls(source)
