# Benchmarking and Analysis of Pre-trained Functionals

This directory contains the scripts and data required to use the pre-trained neural network functionals and reproduce the benchmark results presented in our paper.

The primary analyses include:
1.  **Thermochemical Accuracy:** Evaluating the WTMAD-2 metric on the Diet-GMTKN55 benchmark set.
2.  **Enhancement Factor Analysis:** Plotting the behavior of the enhancement factor `F_xc` to gain physical insight.
3.  **Electron Density Accuracy:** Calculating molecular (avRANE) and atomic (MaxNE) density errors against high-level reference data.

## Prerequisites

- **Environment:** Ensure you have created a virtual environment and installed the necessary packages.
  ```bash
  pip install -r requirements.txt
  ```
- **Pre-trained Models:** The required `.pth` model files are already included in the `DFT/checkpoints/` subdirectory.
- **External Dependencies:** This workflow relies on scripts and data from the original **DietGMTKN55** repository. You must manually add them to this directory.
- **Computational Chemistry Software:** Some analysis steps, particularly for density accuracy, require external tools like **Multiwfn**.

---

## Benchmark 1: Thermochemical Accuracy (Diet-GMTKN55)

This procedure calculates reaction energies for the 30-reaction subset of GMTKN55 and computes the WTMAD-2 error metric.

#### Step 1: Obtain Benchmark Scripts and Data
Before running calculations, you need to download the necessary components from the [DietGMTKN55 repository](https://github.com/gambort/DietGMTKN55).

1.  Download the following files and folders from `https://github.com/gambort/DietGMTKN55/tree/master`:
    - The script `InterfaceG16.py` (this will be used for **analysis only**).
    - The entire `GIF/` directory.
    - The entire `GoodSamples/` directory.

2.  Place them directly into this `test_models/` directory. Your folder structure should look like this:
    ```
    test_models/
    ├── InterfaceG16.py         # <-- Copied script for analysis
    ├── GIF/                    # <-- Copied directory
    ├── GoodSamples/            # <-- Copied directory
    ├── calculate_system_energies.py # <-- Your script for submitting jobs
    ├── DFT/
    ├── Results/
    └── ... (other files)
    ```

#### Step 2: Prepare and Run Energy Calculations

This is a two-part process. First, you generate the necessary input and job script files. Then, you submit the calculation jobs to your SLURM cluster.

##### 2.1 Generate Job Scripts

Run the following two commands in order. The first command creates the system directories and geometry files, and the second populates them with the SLURM scripts required for each functional.

```bash
python InterfaceG16.py --Mode GE
python calculate_system_energies.py --Mode GE
```

##### 2.2 Submit Calculation Jobs

Use the `calculate_system_energies.py` script to submit all calculation jobs for a specific functional to the SLURM queue.

```bash
# Example for our primary model, NN-PBE
python calculate_system_energies.py --Mode CE --Functional NN_PBE_067

# Example for the parent PBE functional
python calculate_system_energies.py --Mode CE --Functional PBE

# Repeat this command for all other functionals you wish to benchmark
# (e.g., NN_PBE_star, NN_XALPHA_067, SCAN, etc.)
```
This will submit a series of jobs. Wait for all of them to complete before proceeding to the next step.

#### Step 3: Collate and Analyze Results
After all jobs are complete, a `.txt` file for each functional will be created in the `Results/` directory, containing the calculated energy for each system.

To compute the WTMAD-2 metric and generate a comparative summary table, use the `InterfaceG16.py` script you downloaded, but now only for its analysis capabilities.

```bash
# 1. Run the analysis mode for each functional to print WTMAD-2 to the console
#    and save the detailed error list.
python InterfaceG16.py --Functional NN_PBE_067 > Results/NN_PBE.txt
python InterfaceG16.py --Functional PBE > Results/PBE-D3BJ.txt
# ... repeat for all other functionals ...

# 2. Convert the text outputs into a single summary CSV file
cd Results/
python txt_to_csv.py
```
This will create `DietGMTKN55_results.csv` in the `Results/` folder, containing a side-by-side comparison of the errors for all evaluated functionals.

---

## Benchmark 2: Enhancement Factor Analysis

This script reproduces the enhancement factor analysis from Figure 4 of the paper by calculating the dependency of `F_xc` on the reduced density gradient `s`.

```bash
python plot_exc.py
```
This will generate `Results/exc.npy`, containing the data used for the plots.

---

## Benchmark 3: Electron Density Accuracy

This is a multi-step process that requires **PySCF**, **Multiwfn**, and reference density files.

### Part A: Molecular Density Accuracy (avRANE)

#### Setup
1.  **Edit Paths:** Open `get_molden.py` and modify the hardcoded paths to your **Multiwfn** executable and the `den_mol_or`/`denrho` directories.
2.  **Reference Densities:** You will need reference CCSD densities. Place them in a file named `den_mol_or/REF/CCSORT.npz`. You will also need LDA densities, which should be placed in the `den_mol_or/calc/LDA_pyscf/` directory (create it if it doesn't exist).

#### Step 1: Run SCF Calculations
This script submits SLURM jobs to run self-consistent field (SCF) calculations for a set of small molecules and atoms, generating `.wfn` files and density grid data.

```bash
# Example for NN-PBE
python run_molden.py --Functional NN_PBE_067
```

#### Step 2: Process Densities and Calculate avRANE
Once the jobs are complete, navigate to the `den_mol_or` directory and run the processing scripts.

```bash
cd ../den_mol_or/

# 1. Convert Multiwfn outputs into compressed .npz files
python calcden.py

# 2. Calculate the Normalized Integral Absolute Difference and avRANE
python dniad
```
The final avRANE values will be printed to the console.

### Part B: Atomic Density Accuracy (MaxNE)

This benchmark uses the atomic `.wfn` files generated in the previous step.

#### Step 1: Generate Radial Distribution Functions (RDFs)
For each functional, navigate to its output directory within `../denrho/dtestin/` and execute the `swfn` script. This script uses Multiwfn to generate `.rdf` files.

```bash
# Example for NN-PBE
cd ../denrho/dtestin/NN_PBE_067/
./swfn
```
*Note: Ensure the path to Multiwfn in the `swfn` script is correct for your system.*

#### Step 2: Calculate MaxNE
Navigate to the `denrho` directory and run the `krms` script to compare the generated RDFs against the CCSD reference and compute the MaxNE metric.

```bash
cd ../../denrho/

# Example for NN-PBE
./krms NN_PBE_067 CCSD
```
The script will print the MaxNE values for RHO, GRD, and LR descriptors to the console.
