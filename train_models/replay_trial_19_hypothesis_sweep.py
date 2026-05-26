import argparse
import json
import pickle
from pathlib import Path

from optuna_joint import (
    DEFAULT_MRKS_DISPERSIONS,
    init_distributed,
    load_chk,
    load_mrks_dispersions,
    run_or_reuse_preoptimization,
    run_trial,
    set_random_seed,
)


HYPOTHESES = {
    "h1_slow_vxc_decay": {
        "lr_train": 2.4e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.5,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 100,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "epoch_schedule": [
            {"name": "vxc_anchor", "start_epoch": 1, "end_epoch": 100, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.5, "vxc_grad_clip": 3.0, "vxc_loss_scale": 100, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "joint_repair", "start_epoch": 101, "end_epoch": 220, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.75, "vxc_grad_clip": 2.0, "vxc_loss_scale": 60, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "balanced_drive", "start_epoch": 221, "end_epoch": 360, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 35, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "fchem_drive", "start_epoch": 361, "end_epoch": 500, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "finish", "start_epoch": 501, "end_epoch": 650, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 10, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
        ],
    },
    "h2_exc_repair_bridge": {
        "lr_train": 2.8e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.6,
        "vxc_grad_clip": 5.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "epoch_schedule": [
            {"name": "clip_drive", "start_epoch": 1, "end_epoch": 72, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.6, "vxc_grad_clip": 5.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "joint_repair", "start_epoch": 73, "end_epoch": 170, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 0.9, "vxc_grad_clip": 2.0, "vxc_loss_scale": 50, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "exc_repair", "start_epoch": 171, "end_epoch": 280, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.75, "vxc_grad_clip": 2.0, "vxc_loss_scale": 35, "exc_loss_scale": 3.0, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "balanced_drive", "start_epoch": 281, "end_epoch": 420, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "finish", "start_epoch": 421, "end_epoch": 600, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 10, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
        ],
    },
    "h3_strong_vxc": {
        "lr_train": 2.0e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.4,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 150,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "epoch_schedule": [
            {"name": "strong_vxc_anchor", "start_epoch": 1, "end_epoch": 140, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.4, "vxc_grad_clip": 2.0, "vxc_loss_scale": 150, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "vxc_joint", "start_epoch": 141, "end_epoch": 300, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.7, "vxc_grad_clip": 2.0, "vxc_loss_scale": 100, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "balanced_drive", "start_epoch": 301, "end_epoch": 460, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 60, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "fchem_drive", "start_epoch": 461, "end_epoch": 600, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 30, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "finish", "start_epoch": 601, "end_epoch": 700, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 20, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
        ],
    },
    "h4_wide_6x48": {
        "lr_train": 1.8e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.5,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 100,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "epoch_schedule": [
            {"name": "vxc_anchor", "start_epoch": 1, "end_epoch": 100, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.5, "vxc_grad_clip": 3.0, "vxc_loss_scale": 100, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "joint_repair", "start_epoch": 101, "end_epoch": 220, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_scale": 0.8, "vxc_grad_clip": 2.0, "vxc_loss_scale": 60, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "balanced_drive", "start_epoch": 221, "end_epoch": 400, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 30, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "finish", "start_epoch": 401, "end_epoch": 650, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 15, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
        ],
    },
    "h5_conflict_clipped": {
        "lr_train": 3.0e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 300.0,
        "reaction_grad_scale": 0.5,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": 3.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "clipped_anchor", "start_epoch": 1, "end_epoch": 100, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.5, "vxc_grad_clip": 3.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.0, "exc_grad_clip": 3.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "clipped_joint", "start_epoch": 101, "end_epoch": 240, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.8, "vxc_grad_clip": 2.0, "vxc_loss_scale": 50, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "clipped_drive", "start_epoch": 241, "end_epoch": 420, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 25, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "finish", "start_epoch": 421, "end_epoch": 600, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 10, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
        ],
    },
    "h6_h5_late_exc_lock": {
        "lr_train": 2.6e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 300.0,
        "reaction_grad_scale": 0.45,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": 3.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "h5_anchor", "start_epoch": 1, "end_epoch": 120, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.45, "vxc_grad_clip": 3.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.0, "exc_grad_clip": 3.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "h5_joint", "start_epoch": 121, "end_epoch": 280, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.75, "vxc_grad_clip": 2.0, "vxc_loss_scale": 50, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "h5_drive", "start_epoch": 281, "end_epoch": 460, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 25, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "exc_lock", "start_epoch": 461, "end_epoch": 620, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.8, "vxc_grad_clip": 1.0, "vxc_loss_scale": 10, "exc_loss_scale": 3.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "locked_finish", "start_epoch": 621, "end_epoch": 800, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 8, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
        ],
    },
    "h7_h5_early_exc_floor": {
        "lr_train": 2.4e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 300.0,
        "reaction_grad_scale": 0.45,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 60,
        "exc_loss_scale": 2.0,
        "exc_grad_clip": 1.5,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "exc_floor_anchor", "start_epoch": 1, "end_epoch": 100, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.45, "vxc_grad_clip": 3.0, "vxc_loss_scale": 60, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "exc_floor_joint", "start_epoch": 101, "end_epoch": 240, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.75, "vxc_grad_clip": 2.0, "vxc_loss_scale": 45, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "exc_floor_drive", "start_epoch": 241, "end_epoch": 420, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 25, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "balanced_repair", "start_epoch": 421, "end_epoch": 600, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
            {"name": "finish", "start_epoch": 601, "end_epoch": 750, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 8, "exc_loss_scale": 1.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
        ],
    },
    "h8_ultra_vxc_exc_projection": {
        "lr_train": 2.0e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 200.0,
        "reaction_grad_scale": 0.25,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 120,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "ultra_vxc_anchor", "start_epoch": 1, "end_epoch": 180, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 200.0, "reaction_grad_scale": 0.25, "vxc_grad_clip": 2.0, "vxc_loss_scale": 120, "exc_loss_scale": 1.0, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "ultra_vxc_joint", "start_epoch": 181, "end_epoch": 360, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.6, "vxc_grad_clip": 1.5, "vxc_loss_scale": 80, "exc_loss_scale": 1.5, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "balanced_drive", "start_epoch": 361, "end_epoch": 540, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 35, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "exc_projection", "start_epoch": 541, "end_epoch": 720, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.7, "vxc_grad_clip": 1.0, "vxc_loss_scale": 15, "exc_loss_scale": 4.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "finish", "start_epoch": 721, "end_epoch": 900, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 10, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h9_low_decay_h5": {
        "lr_train": 2.2e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 300.0,
        "reaction_grad_scale": 0.4,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.5,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "low_decay_anchor", "start_epoch": 1, "end_epoch": 120, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.4, "vxc_grad_clip": 3.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "low_decay_joint", "start_epoch": 121, "end_epoch": 300, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.8, "vxc_grad_clip": 2.0, "vxc_loss_scale": 50, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "low_decay_drive", "start_epoch": 301, "end_epoch": 500, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.5, "vxc_loss_scale": 25, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "exc_repair", "start_epoch": 501, "end_epoch": 650, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 3.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
            {"name": "finish", "start_epoch": 651, "end_epoch": 800, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 8, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
        ],
    },
    "h10_wide_h5_discipline": {
        "lr_train": 1.4e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 200.0,
        "reaction_grad_scale": 0.35,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.5,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "wide_anchor", "start_epoch": 1, "end_epoch": 140, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 200.0, "reaction_grad_scale": 0.35, "vxc_grad_clip": 2.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_joint", "start_epoch": 141, "end_epoch": 320, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.7, "vxc_grad_clip": 1.5, "vxc_loss_scale": 50, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_drive", "start_epoch": 321, "end_epoch": 540, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 25, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_exc_repair", "start_epoch": 541, "end_epoch": 720, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.9, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 3.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_finish", "start_epoch": 721, "end_epoch": 900, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 8, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h11_wide_h8_guarded": {
        "lr_train": 1.8e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 200.0,
        "reaction_grad_scale": 0.30,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 100,
        "exc_loss_scale": 1.5,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "wide_guard_anchor", "start_epoch": 1, "end_epoch": 160, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 200.0, "reaction_grad_scale": 0.30, "vxc_grad_clip": 2.0, "vxc_loss_scale": 100, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_guard_joint", "start_epoch": 161, "end_epoch": 340, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.65, "vxc_grad_clip": 1.5, "vxc_loss_scale": 65, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_guard_drive", "start_epoch": 341, "end_epoch": 540, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 30, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_guard_exc", "start_epoch": 541, "end_epoch": 720, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 0.85, "vxc_grad_clip": 1.0, "vxc_loss_scale": 15, "exc_loss_scale": 3.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
            {"name": "wide_guard_finish", "start_epoch": 721, "end_epoch": 900, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h12_wide_exc_guard": {
        "lr_train": 1.6e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 200.0,
        "reaction_grad_scale": 0.30,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 2.0,
        "exc_grad_clip": 1.5,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "wide_exc_anchor", "start_epoch": 1, "end_epoch": 140, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 200.0, "reaction_grad_scale": 0.30, "vxc_grad_clip": 2.0, "vxc_loss_scale": 75, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_exc_joint", "start_epoch": 141, "end_epoch": 320, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.65, "vxc_grad_clip": 1.5, "vxc_loss_scale": 50, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_exc_drive", "start_epoch": 321, "end_epoch": 520, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 25, "exc_loss_scale": 3.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_exc_repair", "start_epoch": 521, "end_epoch": 700, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 0.85, "vxc_grad_clip": 1.0, "vxc_loss_scale": 15, "exc_loss_scale": 4.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
            {"name": "wide_exc_finish", "start_epoch": 701, "end_epoch": 900, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 2.5, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h13_wide_h4_fchem_tail": {
        "lr_train": 1.8e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": "none",
        "reaction_grad_scale": 0.50,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 100,
        "exc_loss_scale": 1.0,
        "exc_grad_clip": "none",
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "sum",
        "epoch_schedule": [
            {"name": "h4_anchor", "start_epoch": 1, "end_epoch": 100, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 0.50, "vxc_grad_clip": 3.0, "vxc_loss_scale": 100, "exc_loss_scale": 1.0, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "h4_joint", "start_epoch": 101, "end_epoch": 240, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 0.80, "vxc_grad_clip": 2.0, "vxc_loss_scale": 60, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "h4_balanced_drive", "start_epoch": 241, "end_epoch": 430, "params": {"accum_iter": 2, "gradient_merge_strategy": "sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 2.0, "vxc_loss_scale": 30, "exc_loss_scale": 1.5, "exc_grad_clip": "none", "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "sum"}},
            {"name": "clipped_fchem_tail", "start_epoch": 431, "end_epoch": 650, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 18, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
            {"name": "guarded_finish", "start_epoch": 651, "end_epoch": 850, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h14_wide_low_decay_fchem": {
        "lr_train": 2.0e-4,
        "accum_iter": 3,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 300.0,
        "reaction_grad_scale": 0.40,
        "vxc_grad_clip": 3.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.5,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "wide_low_decay_anchor", "start_epoch": 1, "end_epoch": 120, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.40, "vxc_grad_clip": 3.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_low_decay_joint", "start_epoch": 121, "end_epoch": 300, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.80, "vxc_grad_clip": 2.0, "vxc_loss_scale": 50, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_low_decay_drive", "start_epoch": 301, "end_epoch": 500, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.5, "vxc_loss_scale": 25, "exc_loss_scale": 2.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wide_low_decay_repair", "start_epoch": 501, "end_epoch": 670, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 3.0, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0}},
            {"name": "wide_low_decay_finish", "start_epoch": 671, "end_epoch": 850, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 10, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
    "h15_wider_h8_guarded": {
        "lr_train": 1.2e-4,
        "accum_iter": 4,
        "gradient_merge_strategy": "clip_then_sum",
        "reaction_grad_clip": 200.0,
        "reaction_grad_scale": 0.30,
        "vxc_grad_clip": 2.0,
        "vxc_loss_scale": 75,
        "exc_loss_scale": 1.5,
        "exc_grad_clip": 2.0,
        "exc_grad_scale": 1.0,
        "exc_gradient_merge_strategy": "clip_then_sum",
        "epoch_schedule": [
            {"name": "wider_anchor", "start_epoch": 1, "end_epoch": 160, "params": {"accum_iter": 4, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 200.0, "reaction_grad_scale": 0.30, "vxc_grad_clip": 2.0, "vxc_loss_scale": 75, "exc_loss_scale": 1.5, "exc_grad_clip": 2.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wider_joint", "start_epoch": 161, "end_epoch": 340, "params": {"accum_iter": 3, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 300.0, "reaction_grad_scale": 0.65, "vxc_grad_clip": 1.5, "vxc_loss_scale": 50, "exc_loss_scale": 2.0, "exc_grad_clip": 1.5, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wider_drive", "start_epoch": 341, "end_epoch": 540, "params": {"accum_iter": 2, "gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": 500.0, "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 25, "exc_loss_scale": 2.5, "exc_grad_clip": 1.0, "exc_grad_scale": 1.0, "exc_gradient_merge_strategy": "clip_then_sum"}},
            {"name": "wider_exc_repair", "start_epoch": 541, "end_epoch": 720, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 0.85, "vxc_grad_clip": 1.0, "vxc_loss_scale": 15, "exc_loss_scale": 3.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
            {"name": "wider_finish", "start_epoch": 721, "end_epoch": 900, "params": {"accum_iter": 2, "reaction_gradient_merge_strategy": "sum", "vxc_gradient_merge_strategy": "clip_then_sum", "exc_gradient_merge_strategy": "clip_then_sum", "reaction_grad_clip": "none", "reaction_grad_scale": 1.0, "vxc_grad_clip": 1.0, "vxc_loss_scale": 12, "exc_loss_scale": 2.0, "exc_grad_clip": 0.75, "exc_grad_scale": 1.0}},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hypothesis", required=True, choices=sorted(HYPOTHESES))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shared-preopt-checkpoint", type=str, default=None)
    parser.add_argument("--force-preopt", action="store_true")
    parser.add_argument("--name", type=str, default="PBE-LGxGc_6_32")
    parser.add_argument("--model-type", type=str, default="base", choices=["base", "log"])
    parser.add_argument("--n-predopt", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=650)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vxc-batch-size", type=int, default=1)
    parser.add_argument("--lr-predopt", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-vxc", type=int, default=2)
    parser.add_argument("--preopt-vxc-weight", type=float, default=0.0)
    parser.add_argument("--preopt-vxc-steps", type=int, default=0)
    parser.add_argument("--preopt-vxc-target", type=str, default="pbe", choices=["pbe"])
    parser.add_argument("--trial-number", type=int, default=19)
    parser.add_argument("--train-fchem-target", type=float, default=30.0)
    parser.add_argument("--val-vxc-target", type=float, default=0.13)
    parser.add_argument("--val-fchem-soft-cap", type=float, default=90.0)
    parser.add_argument("--save-selected-checkpoints", action="store_true", default=True)
    parser.add_argument("--include-mrks-dispersion", action="store_true")
    parser.add_argument("--mrks-dispersions-pickle", type=str, default=str(DEFAULT_MRKS_DISPERSIONS))
    parser.add_argument("--no-reaction-dispersion", action="store_true")
    return parser.parse_args()


def select_last_epoch(epoch_history):
    if not epoch_history:
        raise ValueError("Cannot select the last epoch from empty history.")
    return epoch_history[-1]


def last_epoch_checkpoint_key(row):
    return (-int(row["epoch"]),)


def main() -> None:
    args = parse_args()
    params = HYPOTHESES[args.hypothesis]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, world_size, device, rank0 = init_distributed()
    set_random_seed(args.seed + args.trial_number)

    if args.no_reaction_dispersion:
        dispersions = {}
    else:
        with (Path(__file__).resolve().parent / "dispersions" / "dispersions.pickle").open("rb") as handle:
            dispersions = pickle.load(handle)
    mrks_dispersions = load_mrks_dispersions(args.mrks_dispersions_pickle) if args.include_mrks_dispersion else None

    data_predopt, data_train, data_val, data_vxc_train, data_vxc_val = load_chk(path=args.checkpoints_dir)
    shared_preopt_checkpoint = run_or_reuse_preoptimization(
        args=args,
        output_dir=output_dir,
        data_predopt=data_predopt,
        data_vxc_train=data_vxc_train,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        rank0=rank0,
    )

    result = run_trial(
        trial_number=args.trial_number,
        params=params,
        args=args,
        shared_preopt_checkpoint=Path(shared_preopt_checkpoint),
        data_train=data_train,
        data_val=data_val,
        data_vxc_train=data_vxc_train,
        data_vxc_val=data_vxc_val,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        dispersions=dispersions,
        mrks_dispersions=mrks_dispersions,
        output_dir=output_dir,
        rank0=rank0,
        epoch_selector=select_last_epoch,
        checkpoint_row_key=last_epoch_checkpoint_key,
    )

    if rank0:
        final_epoch = select_last_epoch(result["epoch_history"])
        print(f"Replay complete for Trial 19 hypothesis {args.hypothesis}.")
        print(f"Final selected epoch: {final_epoch['epoch']}")
        print(
            "Final metrics: "
            f"train_fchem={float(final_epoch['train_fchem']):.8f}, "
            f"val_vxc={float(final_epoch['val_vxc']):.8f}, "
            f"val_exc={float(final_epoch['val_exc']):.8f}, "
            f"val_fchem={float(final_epoch['val_fchem']):.8f}, "
            f"phase={final_epoch.get('phase_name')}"
        )
        print(f"Selected checkpoint: {result.get('selected_checkpoint_path')}")
        print(f"History path: {result.get('history_path')}")
        print(f"Params: {json.dumps(params, sort_keys=True)}")


if __name__ == "__main__":
    main()
