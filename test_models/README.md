# Benchmarking of the functionals on Diet GMTKN55 (30 reactions), enhancement factor calculations


## Preparing data

1) Create and activate virtual environment, install packages from requirements.txt
2) Generate GIF-files:
```
python -m InterfaceG16 --Mode GE
```

## Calculating system energies
1) Calculate NN functionals' energies:
```
python -m InterfaceG16 --Mode CE --Functional {name of the functional}
```
2) Calculate WTMAD-2 and create a .csv table in the Results folder for comparison:
```
python -m InterfaceG16 --Functional {name of the NN_PBE functional} > Results/NN_PBE.txt
python -m InterfaceG16 --Functional {name of the NN_PBE_star functional} > Results/NN_PBE_star.txt
python -m InterfaceG16 --Functional {name of the NN_XALPHA functional} > Results/NN_XAlpha.txt
python -m InterfaceG16 --Functional PBE_D3BJ > Results/PBE-D3BJ.txt
python -m InterfaceG16 --Functional XAlpha > Results/XAlpha.txt
cd Results/
python -m txt_to_csv
```

## Enhancement factor calculations
To calculate the dependency of the enhancement factors on normed gradient, run:
```
python -m plot_exc
```

To calculate the enhancement factor in the molecule of Argon dimer, run:
```
python -m gen_cubgrid_Ar2
```

## Electron and atomic density accuracy calculations
Change the paths to den_mol_or folder, denrho folder and your Multiwfn file in the get_molden.py script, create calc folder and paste the LDA densities there. Create grids folder and paste the grids into it. Paste CCSD reference densities into den_mol_or/REF/CCSORT.npz file.
To calculate the avRANE, run:
```
python -m run_molden --Functional {name of the functional}
```
Then, go to den_mol_or folder and run:
```
python calcden.py calc 128
```
```
python dniad
```
To calculate the MaxNE, run in the denrho/dtestin/{functional name} folder:
```
./sfwn
```
After it's finished, go to denrho folder and run:
```
./krms {functional name} CCSD
```

