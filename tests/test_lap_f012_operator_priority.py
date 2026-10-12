import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import lap_b8_adamw as b8
import lap_f012_operator_priority as f012


def test_only_the_operator_coefficient_is_doubled():
    values = f012.lambdas_f012()
    assert values['op'] == 2.0 * b8.LAMBDAS['op']
    assert values['relchem'] == b8.LAMBDAS['relchem']
    assert values['ae17'] == b8.LAMBDAS['ae17']
    assert values['exc'] == b8.LAMBDAS['exc']
    assert tuple(values) == b8.TASKS


def test_training_does_not_read_clean28_before_the_arm_finishes():
    source = Path(f012.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    assert 'validation' not in ast.get_source_segment(source, train)
    assert 'diet30' not in source
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    segment = ast.get_source_segment(source, execute)
    assert segment.index('train(') < segment.index('evaluate_clean28(')
    assert 'adamw_step' not in ast.get_source_segment(source, next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'preflight'))
