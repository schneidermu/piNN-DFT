# DFT functionals RHO/GRD/LR test on atoms

## Usage for analysis
### Run
```bash
./krms [runctional name] [Ref name]
```
Example:
```bash
./krms PBE0 CCSD
```

## Usage for external wfn files
 * Change path accordingly in content/swfn, copy genGRD.txt, genLR.txt, genRHO.txt, swfn in dtestin/[method] folder
 * Run ./swfn in dtestin/[method] folder
 * Analyse
