# Incorporating Scientific Knowledge into Neural Network Density Functionals

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

This repository contains the official implementation and pre-trained models for the paper: **"Incorporating scientific knowledge into neural network density functionals"**.

Our work presents a novel, physics-informed approach to constructing accurate and robust density functionals by fusing machine learning with foundational physical principles. Instead of training a "black-box" model from scratch, we train a neural network to perform an exact-constraints-aware, local density-guided fine-tuning of the parameters within the physically-sound Perdew-Burke-Ernzerhof (PBE) functional form.

The resulting functional, **NN-PBE-D3(BJ)**, reduces the thermochemical error of its parent PBE by nearly 30%, achieving the accuracy of modern meta-GGAs while maintaining excellent physical fidelity in electron densities.

## Table of Contents
- [Models Overview](#models-overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [Reproducing Results](#reproducing-results)
- [How to Cite](#how-to-cite)
- [Authors](#authors)
- [License](#license)

## Models Overview

This repository provides code and pre-trained weights for the four main functionals discussed in the paper:

- **`NN-PBE-D3(BJ)`**: Our primary, physics-informed model that satisfies all original PBE constraints. This model demonstrates the best balance of thermochemical accuracy and physical soundness.
- **`NN-PBE*-D3(BJ)`**: An ablation study model trained *without* enforcing the analytical constraints of PBE. Used to quantify the importance of constraint satisfaction.
- **`NN-PBE**-D3(BJ)`**: An ablation study model that follows the approach of Nagai et al., where the NN predicts corrective enhancement factors rather than modifying internal parameters.
- **`NN-Xα-D3(BJ)`**: A baseline model trained on the much simpler Xα functional form to highlight the benefits of using a physically-grounded scaffold like PBE.

Pre-trained model weights are available in the [`test_models/DFT/checkpoints/`](test_models/DFT/checkpoints/) directory.

## Repository Structure

The repository is organized into several key directories:

```
└── piNN-DFT/
    ├── train_models/         # Scripts for training the NN functionals from scratch.
    ├── test_models/          # Scripts for benchmarking, analysis, and using pre-trained models.
    ├── dft_functionals/      # Core PyTorch implementations of the baseline DFT functionals (PBE, SVWN3).
    ├── MN_dataset/           # Metadata and download instructions for the training dataset.
    ├── den_mol_or/           # Scripts for analyzing molecular electron density accuracy (avRANE).
    └── denrho/               # Scripts for analyzing atomic electron density accuracy (MaxNE).
```

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/schneidermu-pinn-dft.git
    cd schneidermu-pinn-dft
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The required packages are listed in `requirements.txt` files within the `train_models` and `test_models` directories. For most use cases, the testing requirements are sufficient.

    ```bash
    # For benchmarking and using pre-trained models
    pip install -r test_models/requirements.txt

    # For training new models from scratch
    pip install -r train_models/requirements.txt
    ```
    *Note: The Nagai et al. model (`pcNN_mol`) has separate dependencies listed in `test_models/requirements_nagai.txt`.*

4.  **Download the Dataset**
    The training and local energy data are required for both training and parts of the analysis. Download the dataset from the link provided in the [`MN_dataset/README.md`](MN_dataset/README.md) and place the `.h5` files in a `data/` directory inside `train_models/`.

## Usage

This repository supports both training new functionals and using our pre-trained models for calculations and benchmarking. As per the modular structure, detailed instructions are located within the relevant directories.

#### 1. Training a New Functional
To train the models from scratch, navigate to the `train_models` directory. The process involves preparing the downloaded dataset and then running the main training script.

Detailed instructions are provided in **[`train_models/README.md`](train_models/README.md)**.

#### 2. Benchmarking and Inference
To perform calculations or reproduce the benchmark results (e.g., Diet-GMTKN55) using our pre-trained models, navigate to the `test_models` directory. This folder contains the necessary scripts to evaluate system energies, calculate enhancement factors, and assess density accuracies.

- **Pre-trained models** are located in [`test_models/DFT/checkpoints/`](test_models/DFT/checkpoints/).
- Detailed instructions for running benchmarks are in **[`test_models/README.md`](test_models/README.md)**.

#### 3. Electron Density Analysis
Scripts for calculating molecular (avRANE) and atomic (MaxNE) electron density errors are available in the `den_mol_or` and `denrho` directories, respectively. Please refer to the README files within those folders for specific usage.

## Data Availability

The training dataset, which consists of the M06-2X parametrization set with pre-computed local PBE energies, is available for download. Please find the download link and a description of the data structure in the **[`MN_dataset/README.md`](MN_dataset/README.md)** file.

## Reproducing Results

You can reproduce the key findings of our paper using the scripts in the `test_models` directory:
- **Thermochemical Accuracy (Diet-GMTKN55):** Use `script.py` and `InterfaceG16.py` to calculate reaction energies for the 30-reaction subset and obtain the WTMAD-2 metric.
- **Enhancement Factor Plots:** Run `plot_exc.py` to generate the data for Figure 4.
- **Density Accuracy (avRANE & MaxNE):** Use the scripts in `den_mol_or/` and `denrho/` as described in their respective READMEs.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. For the full legal text, see the [LICENSE](LICENSE) file.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
