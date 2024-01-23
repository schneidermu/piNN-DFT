# SCF Calculations for NN-parametrized PBE and XAlpha

## Benchmark on diet GMTKN55 (30 and 50 reactions), enhancement factor for Argon dimer calculation


### Reproducing the experiment:

1) Clone the repository
2) Create and activate virtual environment, install packages from requirements.txt
3) Go to SCF-calculation folder and generate GIF-files:
```
cd SCF-calculations/
python -m InterfaceG16 --Mode GE
```
4) Calculate NN functionals' energies:
```
python -m InterfaceG16 --Mode CE
```

5) Calculate PBE and XAlpha energies:

```
python -m InterfaceG16 --Mode CE --Functional Non-NN
```
6) Calculate PBE0 and PBE D3BJ energies:
```
python -m InterfaceG16 --Mode D3
```

7) Add the D3BJ corrections to PBE and XAlpha energies (for NN functionals they are added during step 4 calculations):

```
python -m add_d3_corrections
```

8) Calculate WTMAD-2 and create a .csv table in the Results folder for comparison:
```
python -m InterfaceG16 --Functional NN_PBE > Results/NN_PBE.txt
python -m InterfaceG16 --Functional NN_XALPHA > Results/NN_XAlpha.txt
python -m InterfaceG16 --Functional PBE_D3BJ > Results/PBE-D3BJ.txt
python -m InterfaceG16 --Functional XAlpha > Results/XAlpha.txt
cd Results/
python -m txt_to_csv
```
9) To reproduce Fig. , run R script
10) To calculate WTMAD-2 for 30 reactions as in Table ., use --NFinal 30 argument for steps 3-8
