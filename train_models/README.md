# Training the Neural Network Functionals

This directory contains all the necessary scripts to train the neural network density functionals (`NN-PBE`, `NN-Xα`, and their variants) from scratch. The process is divided into two main stages: data preparation and model training.

**Note:** The training process is computationally intensive (all functionals were trained on 2X Nvidia V100 for ~10 hours) and is designed to be run on a high-performance computing (HPC) cluster using the SLURM workload manager.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Data Preparation](#step-1-data-preparation)
- [Step 2: Model Training](#step-2-model-training)

## Prerequisites

1.  **Environment:** Ensure you have created a virtual environment and installed all required packages.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset:** Download the complete MN dataset as instructed in the main [`MN_dataset/README.md`](../MN_dataset/README.md).

## Step 1: Data Preparation

This step processes the raw `.h5` files into stratified training and validation sets, which are then saved as pickled dictionaries for efficient loading during training.

1.  **Organize Dataset Files:**
    Create a `data` directory inside the current `train_models/` folder and place all the downloaded `.h5` files into it. The expected structure is:
    ```
    train_models/
    ├── data/
    │   ├── Al_ae17.h5
    │   ├── Al_mgae109.h5
    │   └── ... (all other .h5 files)
    ├── prepare_data.py
    └── ...
    ```

2.  **Run the Preparation Script:**
    Execute the `prepare_data.py` script from the `train_models` directory.
    ```bash
    python prepare_data.py
    ```

3.  **Expected Outcome:**
    The script will create a `checkpoints` directory containing three files:
    - `data.pickle`: The complete, processed dataset.
    - `data_train.pickle`: The training subset (80%).
    - `data_test.pickle`: The validation subset (20%).

## Step 2: Model Training

The training workflow is managed by `calculations.py`, which acts as a launcher for submitting a series of SLURM jobs. Each job trains a model with a specific hyperparameter configuration, primarily sweeping through different values for `Ω` (omega) as described in the paper. The core training logic is implemented in `predopt_train.py`.

1.  **Configure Training Jobs (Optional):**
    The `calculations.py` script is pre-configured to replicate the hyperparameter search performed in our paper. You can modify this script to change the range of `Ω`, adjust the neural network architecture (`functionals` list), or alter other training parameters.

2.  **Launch Training Jobs:**
    Execute the script to submit the jobs to your SLURM cluster.
    ```bash
    python calculations.py
    ```

3.  **Expected Outcome:**
    - A series of SLURM jobs will be submitted to the queue.
    - Training logs will be written to the `train_models/logs/` directory.
    - Model checkpoints will be saved periodically during training to the `train_models/best_models/` directory, with filenames indicating the model, epoch, and performance metrics.
