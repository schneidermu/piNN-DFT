# MN Neuromorphic dataset

## Dataset structure
Each h5 file corresponds to one substance.

h5 file contain two HDF5 datasets: `ener` and `grid`.

- `ener` (3 float values)
    1.  Kinetic and Potential energy
    2.  HF exchange energy
    3.  Total PBE0 energy

- `grid` (Nx12 matrix where N is number of grid points and 12 is number of features)
    1. x coordinate
    2. y coordinate
    3. z coordinate
    4. Integration weight
    5. Alpha electron density
    6. Beta electron density
    7. Alpha alpha contracted gradient
    8. Alpha beta contracted gradient
    9. Beta beta contracted gradient
    10. Alpha kinetic energy density
    11. Beta kinetic energy density
    12. Local HF exchange energy
    
`Reference_data.csv` - reference energies

`total_dataframe_sorted_final.csv` - reactions coefficients

## Download
OneDrive dowload link - [DOWNLOAD](https://1drv.ms/u/s!AgdbSNqMtbhRgcYREkBwLcfoyQ1CNg?e=AIU3T8)
