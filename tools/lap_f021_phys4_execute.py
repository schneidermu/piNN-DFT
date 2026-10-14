"""Execution guard: hard whole-workflow wall budget and training-only metadata."""
import argparse
import sys
import time

import lap_f021_phys4 as exp
import lap_f021_phys4_eval as evaluation
import lap_f021_phys4_run as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('parity', 'train', '64', '128', 'clean'))
    args = parser.parse_args()
    contract_path = exp.ROOT / 'lap_f021_phys4_execution_contract.json'
    if contract_path.exists():
        contract = exp.read('execution_contract')
        assert contract['driver_sha256'] == exp.audit.sha(__file__)
        assert contract['protocol_sha256'] == exp.audit.sha(exp.ROOT / 'lap_f021_phys4_protocol.json')
    else:
        origin = min((exp.ROOT / 'lap_f021_phys4_protocol.json').stat().st_mtime,
                     (exp.OUT / 'draft_protocol_before_scientific_execution.json').stat().st_mtime)
        contract = {'driver_sha256': exp.audit.sha(__file__),
                    'protocol_sha256': exp.audit.sha(exp.ROOT / 'lap_f021_phys4_protocol.json'),
                    'wall_clock_origin_unix': origin, 'budget_seconds': exp.BUDGET,
                    'training_reader': 'qualified TrainingBundle; no validation indices',
                    'purpose': 'execution isolation only; unchanged scientific protocol and producers'}
        exp.write('execution_contract', contract)
    original_check = exp.Budget.check

    def check(self, reserve=0):
        wall = time.time() - contract['wall_clock_origin_unix']
        self.prior = max(self.prior, wall - (time.perf_counter() - self.start))
        return original_check(self, reserve)

    exp.Budget.check = check
    training_module = exp.b8._import_run()
    original_reader = training_module.PublicationDataset
    if args.stage == 'train':
        assert exp.read('source_zero_parity')['passed']
        training_module.PublicationDataset = exp.audit.TrainingBundle
    try:
        budget = exp.Budget()
        budget.check(600)
        module = run if args.stage in ('parity', 'train') else evaluation
        sys.argv = [module.__file__, args.stage]
        module.main()
    finally:
        training_module.PublicationDataset = original_reader
        exp.Budget.check = original_check


if __name__ == '__main__':
    main()
