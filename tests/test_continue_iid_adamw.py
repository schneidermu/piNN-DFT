from pathlib import Path

import pytest
import torch

from tools import continue_iid_adamw as continuation


def test_resume_preserves_original_and_stops_at_requested_checkpoints(tmp_path, monkeypatch):
    source, output = tmp_path / 'original', tmp_path / 'continuation'
    for root in (source, output):
        (root / 'ordinary_sgd_adamw').mkdir(parents=True)
        torch.save({'cursor': 59}, root / 'ordinary_sgd_adamw/latest.pt')
    before = continuation.run.file_sha256(source / 'ordinary_sgd_adamw/latest.pt')
    monkeypatch.setattr(continuation, 'SOURCE', source)
    monkeypatch.setattr(continuation, 'OUTPUT', output)
    monkeypatch.setattr(continuation, 'START_SHA', before)
    monkeypatch.setattr(continuation, 'prepare', lambda: None)
    steps = []

    def train(folder, total, *, stop_at, **kwargs):
        assert total == 90 and kwargs['constant_lr'] and kwargs['learning_rate'] == 1e-4
        latest = folder / 'ordinary_sgd_adamw/latest.pt'
        current = torch.load(latest, weights_only=False)['cursor']
        steps.extend(range(current, stop_at))
        torch.save({'cursor': stop_at}, latest)

    monkeypatch.setattr(continuation.run, 'train', train)
    continuation.train()
    assert steps == list(range(59, 90))
    for cursor in (70, 80, 90):
        path = output / f'ordinary_sgd_adamw/checkpoint_{cursor}.pt'
        assert torch.load(path, weights_only=False)['cursor'] == cursor
    continuation.train()
    assert steps == list(range(59, 90))  # No replay of completed updates.
    assert continuation.run.file_sha256(source / 'ordinary_sgd_adamw/latest.pt') == before


def test_preflight_rejects_wrong_start_before_loading_model(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'ordinary_sgd_adamw').mkdir(parents=True)
    (source / 'ordinary_sgd_adamw/latest.pt').write_bytes(b'wrong checkpoint')
    monkeypatch.setattr(continuation, 'SOURCE', Path(source))
    with pytest.raises(AssertionError):
        continuation.prepare()
