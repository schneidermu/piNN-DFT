"""Publication-style figures sourced exclusively from F021 numeric tables."""
import matplotlib

matplotlib.use('Agg')
import lap_f021_phys4 as exp
import matplotlib.pyplot as plt
import numpy as np
from lap_f021_phys4_report import csv_rows


def save(fig, name):
    fig.tight_layout()
    fig.savefig(exp.ROOT / ('lap_f021_phys4_' + name + '.png'), dpi=160)
    plt.close(fig)


def main():
    tasks = csv_rows('task_variance')
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tasks))
    ax.bar(x - .18, [float(r['singleton_MSE']) for r in tasks], .36, label='singleton')
    ax.bar(x + .18, [float(r['phys4_MSE']) for r in tasks], .36, label='PHYS4')
    ax.set_xticks(x, [r['task'] for r in tasks])
    ax.set_yscale('log')
    ax.set_ylabel('Raw full-vector gradient MSE')
    ax.legend()
    save(fig, 'task_vector_mse')
    joint = csv_rows('joint_variance')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, r in zip(axes, joint, strict=True):
        ax.bar(['singleton', 'PHYS4'], [float(r['singleton_MSE']), float(r['phys4_MSE'])])
        ax.set_title(r['object'])
        ax.set_ylabel('Fixed-weight full-vector MSE')
    save(fig, 'weighted_and_composite_mse')
    covariance = csv_rows('covariance_decomposition')
    chemical = next(r for r in covariance if r['left'] == 'chemistry')
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, suffix in enumerate(('singleton', 'phys4')):
        physical = float(joint[0][suffix + '_MSE'])
        total = float(joint[1][suffix + '_MSE'])
        cross = 2 * float(chemical[suffix + '_covariance_trace'])
        chem = total - physical - cross
        ax.bar(x - .22, chem, .22, label='chemistry MSE' if x == 0 else None)
        ax.bar(x, physical, .22, label='physical MSE' if x == 0 else None)
        ax.bar(x + .22, cross, .22, label='2 × chemistry/physical covariance' if x == 0 else None)
    ax.set_xticks([0, 1], ['singleton', 'PHYS4'])
    ax.set_ylabel('Composite vector MSE contributions')
    ax.axhline(0, color='gray', linewidth=.7)
    ax.legend(fontsize=8)
    save(fig, 'chemistry_covariance')
    distributions = csv_rows('gradient_distributions')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, task in zip(axes, exp.TASKS, strict=True):
        for label, suffix in [('singleton', '1'), ('PHYS4', '4')]:
            values = np.sort([float(r[task + '_cosine_' + suffix]) for r in distributions])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
        ax.set_title(task)
        ax.set_xlabel('Cosine to empirical prefix mean')
        ax.set_ylabel('Empirical CDF (128 positions)')
        ax.legend()
    save(fig, 'physical_cosine_distributions')
    checkpoint_file = exp.ROOT / 'lap_f021_phys4_checkpoint_metrics.csv'
    if not checkpoint_file.exists():
        return
    rows = csv_rows('checkpoint_metrics')
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, task in zip(axes, exp.b8.TASKS, strict=True):
        ax.plot([int(r['cursor']) for r in rows], [float(r[task + '_P536_ratio']) for r in rows], 'o-')
        ax.axhline(1, color='gray', linestyle='--')
        ax.set_title(task)
        ax.set_xlabel('Optimizer updates')
        ax.set_ylabel('Exact fixed-panel / P536 ratio')
    save(fig, 'full_objective_trajectories')
    db = csv_rows('database_metrics')
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, name in zip(axes.flat, sorted({r['database'] for r in db}), strict=True):
        selected = [r for r in db if r['database'] == name]
        ax.plot([int(r['cursor']) for r in selected], [float(r['mae']) for r in selected], 'o-', label='PHYS4')
        ax.axhline(float(selected[0]['F010_t128_MAE']), linestyle='--', color='gray', label='F010 t128')
        ax.set_title(name)
        ax.set_xlabel('Optimizer updates')
        ax.set_ylabel('MAE (kcal/mol)')
    axes.flat[0].legend()
    save(fig, 'database_chemistry')
    logs_path = exp.OUT / 'training_status.json'
    if logs_path.exists():
        new = exp.audit.read(logs_path)['logs']
        old = exp.audit.read(exp.f010.OUT / 'training_status.json')['logs'][:128]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for label, logs in [('F010', old), ('PHYS4', new)]:
            axes[0].plot([r['local_update'] for r in logs], [r['displacement_from_t128'] for r in logs], label=label)
            axes[1].plot([r['local_update'] for r in logs], [r['update_norm'] for r in logs], label=label)
        axes[0].set_ylabel('Parameter displacement from B8')
        axes[1].set_ylabel('Native AdamW update norm')
        for ax in axes:
            ax.set_xlabel('Optimizer updates')
            ax.legend()
        save(fig, 'adamw_trajectory')
    clean_path = exp.ROOT / 'lap_f021_phys4_clean28_receipt.json'
    if clean_path.exists() and 'clean28' in exp.read('clean28_receipt'):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['F010 t128', 'historical best', 'PHYS4 t128'],
               [8.68988195905555, 8.619694171538775, exp.read('clean28_receipt')['clean28']])
        ax.set_ylabel('Clean28 fixed-density weighted error (kcal/mol)')
        save(fig, 'clean28_comparison')


if __name__ == '__main__':
    main()
