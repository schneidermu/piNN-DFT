"""All-90 preprocessing reuses the qualified builder without selecting a subset."""

import json
import sys

import pytest

import lap_prepare_all90_operator as builder


def test_all90_sorted_exact_builder_contract(tmp_path, monkeypatch):
    central = tmp_path / "central"
    central.mkdir()
    names = [f"system{i:02}" for i in range(90)]
    (central / "manifest.json").write_text(json.dumps({"records": [
        {"system_name": name} for name in reversed(names)]}))
    calls = []
    monkeypatch.setattr(builder, "build_ao_factor_cache", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(sys, "argv", ["builder", "--central", str(central),
                                    "--output", str(tmp_path / "output")])
    builder.main()
    assert calls == [((central, tmp_path / "output"), {"chunk_size": 2048, "systems": tuple(names)})]


@pytest.mark.parametrize("count,duplicate", [(89, False), (90, True), (91, False)])
def test_missing_duplicate_extra_fail_before_generation(tmp_path, monkeypatch, count, duplicate):
    names = [str(i) for i in range(count)]
    if duplicate:
        names[-1] = names[0]
    (tmp_path / "manifest.json").write_text(json.dumps({"records": [
        {"system_name": name} for name in names]}))
    monkeypatch.setattr(builder, "build_ao_factor_cache", lambda *a, **kw: pytest.fail("must not build"))
    monkeypatch.setattr(sys, "argv", ["builder", "--central", str(tmp_path),
                                    "--output", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="90 unique"):
        builder.main()
