import argparse
import csv
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch

import tmp_eval_pbe_vxc_exc as pbe_eval
import tmp_eval_trial19_vxc_exc as nn_eval

HARTREE2KCAL = 627.5095


def load_model(checkpoint, model_name, model_type, device):
    model = nn_eval.build_model(
        SimpleNamespace(name=model_name, dropout=0.0, model_type=model_type),
        device,
    )
    nn_eval.load_state_dict_into_model(model, Path(checkpoint), device)
    model.eval()
    return model


def add_prediction_columns(row, prefix, pred_value, ref):
    res = pred_value - ref
    row[f"{prefix}_exc_ha"] = pred_value
    row[f"{prefix}_residual_ha"] = res
    row[f"{prefix}_residual_kcal"] = res * HARTREE2KCAL
    row[f"{prefix}_abs_residual_kcal"] = abs(res) * HARTREE2KCAL
    row[f"{prefix}_relative_residual"] = res / abs(ref)
    row[f"{prefix}_abs_relative_residual"] = abs(res) / abs(ref)
    row[f"{prefix}_relative_residual_vs_pred"] = res / abs(pred_value)
    row[f"{prefix}_abs_relative_residual_vs_pred"] = abs(res) / abs(pred_value)
    return row


def iter_split(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def prediction_row(model, item, split, device, legacy_model=None, include_mrks_dispersion=False, mrks_dispersions=None):
    batch = pbe_eval.vxc_collate_fn([item])
    _, nn_pred, nn_ref = nn_eval.exc_loss(
        model,
        batch,
        device,
        dispersions=mrks_dispersions,
        include_mrks_dispersion=include_mrks_dispersion,
    )
    _, pbe_pred, pbe_ref = pbe_eval.pbe_exc_loss(
        batch,
        device,
        dispersions=mrks_dispersions,
        include_mrks_dispersion=include_mrks_dispersion,
    )
    legacy_pred_value = None
    if legacy_model is not None:
        _, legacy_pred, legacy_ref = nn_eval.exc_loss(
            legacy_model,
            batch,
            device,
            dispersions=mrks_dispersions,
            include_mrks_dispersion=include_mrks_dispersion,
        )
        legacy_ref_value = float(legacy_ref[0].detach().cpu())
    else:
        legacy_ref_value = None

    ref = float(nn_ref[0].detach().cpu())
    pbe_ref_value = float(pbe_ref[0].detach().cpu())
    if abs(ref - pbe_ref_value) > 1e-6:
        raise RuntimeError(f"Reference mismatch for {item['Name']}: {ref} vs {pbe_ref_value}")
    if legacy_ref_value is not None and abs(ref - legacy_ref_value) > 1e-6:
        raise RuntimeError(f"Legacy reference mismatch for {item['Name']}: {ref} vs {legacy_ref_value}")

    nn_pred_value = float(nn_pred[0].detach().cpu())
    pbe_pred_value = float(pbe_pred[0].detach().cpu())

    row = {
        "split": split,
        "system": item["Name"],
        "ref_exc_ha": ref,
    }
    add_prediction_columns(row, "nn_pbe_l", nn_pred_value, ref)
    add_prediction_columns(row, "pure_pbe", pbe_pred_value, ref)
    if legacy_model is not None:
        legacy_pred_value = float(legacy_pred[0].detach().cpu())
        add_prediction_columns(row, "legacy_nn_pbe", legacy_pred_value, ref)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trial_19_selected.pt")
    parser.add_argument("--name", default="PBE-LGxGc_6_64")
    parser.add_argument("--model-type", choices=["current", "legacy"], default="current")
    parser.add_argument("--legacy-checkpoint", default=None)
    parser.add_argument("--legacy-name", default="legacy_6_32")
    parser.add_argument("--legacy-model-type", choices=["current", "legacy"], default="legacy")
    parser.add_argument("--train-pickle", default="../../checkpoints/data_vxc_train.pickle")
    parser.add_argument("--val-pickle", default="../../checkpoints/data_vxc_val.pickle")
    parser.add_argument("--output", default="exc_residual_analysis/exc_predictions_cache.csv")
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument("--mrks-dispersions-pickle", default=str(nn_eval.DEFAULT_MRKS_DISPERSIONS))
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_model(args.checkpoint, args.name, args.model_type, device)
    legacy_model = None
    if args.legacy_checkpoint:
        legacy_model = load_model(args.legacy_checkpoint, args.legacy_name, args.legacy_model_type, device)
    mrks_dispersions = (
        nn_eval.load_mrks_dispersions(args.mrks_dispersions_pickle)
        if args.include_mrks_dispersion
        else None
    )

    rows = []
    for split, path in (("train", args.train_pickle), ("val", args.val_pickle)):
        for item in iter_split(path):
            rows.append(
                prediction_row(
                    model,
                    item,
                    split,
                    device,
                    legacy_model=legacy_model,
                    include_mrks_dispersion=args.include_mrks_dispersion,
                    mrks_dispersions=mrks_dispersions,
                )
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} systems to {output_path}")


if __name__ == "__main__":
    main()
