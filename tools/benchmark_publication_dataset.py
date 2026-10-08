"""Bounded CPU I/O benchmark; no functional evaluation or scientific gradients."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_models.publication_data import PublicationDataset, identity_collate


class SpeciesView(Dataset):
    def __init__(self, validation, ids):
        self.validation, self.ids = validation, tuple(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.validation.species(self.ids[index])


def ram():
    try:
        import psutil
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except ImportError:
        return None


def measure(dataset, indices):
    samples = Subset(dataset, indices)
    start = time.perf_counter()
    samples[0]
    first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(5):
        samples[0]
    warm = (time.perf_counter() - start)/5
    rates, peak = {}, ram()
    for workers in (0, 2):
        loader = DataLoader(samples, batch_size=1, num_workers=workers, persistent_workers=workers > 0,
                            collate_fn=identity_collate)
        epochs = []
        for _ in range(2):
            start, count = time.perf_counter(), 0
            for batch in loader:
                count += len(batch)
                observed = ram()
                if observed is not None:
                    peak = max(peak or 0, observed)
            epochs.append(count / (time.perf_counter()-start))
        rates[str(workers)] = {"first_epoch_items_per_second": epochs[0], "warm_epoch_items_per_second": epochs[1]}
        del loader
    return {"first_handle_access_seconds_including_integrity_check": first,
            "same_sample_warm_seconds": warm, "dataloader": rates,
            "sampled_peak_host_ram_bytes_parent_and_workers": peak}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for label in ("chemistry_random_variant", "chemistry_sequential", "mrks_energy", "operator_chunks", "validation_species"):
        bundle = PublicationDataset(args.root)
        if label.startswith("chemistry"):
            data = bundle.chemistry()
            if label == "chemistry_sequential":
                indices = list(range(16))
            else:
                from train_models.publication_data.contracts import VARIANTS
                rng = np.random.default_rng(42)
                indices = [(data.ids[int(i)], VARIANTS[int(v)]) for i, v in
                           zip(rng.choice(len(data), 16, replace=False), rng.integers(0, 8, 16))]
        elif label == "mrks_energy":
            data, indices = bundle.mrks(), list(range(16))
        elif label == "operator_chunks":
            data = bundle.mrks().ao_dataset(next(iter(bundle.systems)), 4096)
            indices = list(range(min(16, len(data))))
        else:
            data = SpeciesView(bundle.validation("diet30_diagnostic"), sorted(bundle.validation_species))
            indices = list(range(16))
        results[label] = measure(data, indices)
        bundle.close()
        print(json.dumps({"benchmark_completed": label}), flush=True)
    args.output.write_text(json.dumps({"logical_sha256": bundle.manifest["logical_sha256"], "workloads": results,
        "method": "Fresh worker-local HDF5 handles; first access SHA checks included; OS cache not forcibly purged; persistent workers for second epoch",
        "no_model_evaluation": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
