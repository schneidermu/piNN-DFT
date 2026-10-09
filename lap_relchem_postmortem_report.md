# Relchem postmortem: why the fixed panel stays above P536

This is an offline reconstruction of existing fixed-variant chemistry receipts. No model was evaluated again.

Qualified singleton loss from optuna_joint.batch_fchem: database_weight * frequency_weight / mean_weight * sqrt(MSE + 1e-20). The fixed-panel relchem objective is the equal-identity mean of these 251 scalars. Database and frequency factors were not applied again.

The stored P536 relchem objective is 1.2354710978866068. Recalculated ratios agree with the published nine-decimal receipts to within 5e-10. AE17, Exc, and operator ratios below are copied from the existing audit JSON files and were not recomputed.

| Checkpoint | Relchem | AE17 | Exc | Operator |
|---|---:|---:|---:|---:|
| iid_t59 | 1.013391210 | 0.260040533 | 0.280853918 | 0.967795091 |
| iid_t70 | 1.044607979 | 0.258233903 | 0.196699782 | 0.941692051 |
| iid_t80 | 1.019138724 | 0.341304334 | 0.361633060 | 0.946732495 |
| iid_t90 | 1.026268288 | 0.616606025 | 0.553931114 | 0.939497157 |
| dbstrat_t70 | 1.031209044 | 0.315539432 | 0.322297455 | 0.919694809 |
| dbstrat_t90 | 1.018877965 | 0.194544149 | 0.108227088 | 0.920716660 |
| lr3e-5_t80 | 1.034869578 | 0.080727952 | 0.059058103 | 0.944069705 |

## Finding

The relchem ratio stays above 1 at every audited checkpoint because a minority of identities become much worse in the qualified singleton loss. At IID t70, 198 of 251 identities improve and 53 worsen. The worsening mass is 0.2037 and the net rise is 0.0551. The largest 20 positive reactions account for 79.3% of the worsening mass and 2.9 times the net rise, because the improvements cancel most of the damage.

The net rise is carried by the small databases ABDE4, pTC13, and PA8. At IID t70 their contributions to the change are +0.0592, +0.0586, and +0.0295. Together they exceed the net rise. Databases with a negative delta at every audited checkpoint: DBH76, MGAE109, NCCE31. Databases with a positive delta at every audited checkpoint: ABDE4, PA8, pTC13. Population size is not the source of the regression: the large databases are in the improving group.

The same three small databases stay above P536 under database-stratified sampling. At t70, 43 identities worsen in both streams, and 19 of the 20 largest IID worsenings are also among the 20 largest DB-strat worsenings. DB-strat reduces the ABDE4 delta relative to IID but does not make ABDE4, pTC13, or PA8 better than P536. The failure is therefore not an IID-only omission, and it is not spread evenly over all 251 identities.

## What the objective change is made of

A positive database delta raises the relchem objective. Contribution is the sum of qualified singleton losses in that database divided by 251. Mean loss is the same sum divided by the database size. These are not ordinary unweighted MAE values.

| Checkpoint | Relchem | Ratio to P536 | Net delta | Top-20 share of worsening mass | Identities worsened |
|---|---:|---:|---:|---:|---:|
| iid_t59 | 1.252015550937 | 1.013391210 | +1.654445e-02 | 0.803 | 52 / 251 |
| iid_t70 | 1.290582966818 | 1.044607979 | +5.511187e-02 | 0.793 | 53 / 251 |
| iid_t80 | 1.259116438808 | 1.019138724 | +2.364534e-02 | 0.801 | 55 / 251 |
| iid_t90 | 1.267924808154 | 1.026268288 | +3.245371e-02 | 0.800 | 49 / 251 |
| dbstrat_t70 | 1.274028970110 | 1.031209044 | +3.855787e-02 | 0.842 | 68 / 251 |
| dbstrat_t90 | 1.258794277507 | 1.018877965 | +2.332318e-02 | 0.785 | 72 / 251 |
| lr3e-5_t80 | 1.278551453814 | 1.034869578 | +4.308036e-02 | 0.796 | 55 / 251 |

### Eight-database change from P536

| Database | n | iid_t59 | iid_t70 | iid_t80 | iid_t90 | dbstrat_t70 | dbstrat_t90 | lr3e-5_t80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 4 | +4.3296e-02 | +5.9218e-02 | +5.6267e-02 | +2.3291e-02 | +2.7890e-02 | +1.0884e-02 | +5.8809e-02 |
| DBH76 | 70 | -1.5891e-02 | -2.2432e-02 | -1.9900e-02 | -1.7080e-02 | -1.5069e-02 | -1.3683e-02 | -2.1508e-02 |
| EA13 | 11 | +5.9213e-03 | +6.8503e-03 | +9.4754e-03 | -4.7617e-04 | +4.3350e-03 | -2.3231e-03 | +7.9585e-03 |
| IP13 | 13 | +3.4318e-03 | +1.1367e-02 | -2.5266e-03 | +3.5286e-02 | +2.7724e-03 | +2.1865e-02 | +5.2753e-03 |
| MGAE109 | 104 | -2.7156e-02 | -3.0744e-02 | -3.3275e-02 | -1.5770e-02 | -1.1117e-02 | -2.1255e-03 | -3.1999e-02 |
| NCCE31 | 28 | -4.1894e-02 | -5.7285e-02 | -5.6479e-02 | -3.4786e-02 | -4.1790e-02 | -3.0154e-02 | -5.7226e-02 |
| PA8 | 8 | +1.4102e-02 | +2.9545e-02 | +2.2233e-02 | +1.0394e-02 | +2.0885e-02 | +8.1553e-03 | +2.7000e-02 |
| pTC13 | 13 | +3.4734e-02 | +5.8593e-02 | +4.7851e-02 | +3.1594e-02 | +5.0653e-02 | +3.0705e-02 | +5.4771e-02 |

### IID t70 and DB-strat t70 in the qualified-loss units

| Database | n | IID mean | IID contribution | IID delta | DB mean | DB contribution | DB delta | DB minus IID | IID improved | DB improved | IID positive | IID negative | DB positive | DB negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABDE4 | 4 | 9.3264 | 0.1486 | +5.9218e-02 | 7.3605 | 0.1173 | +2.7890e-02 | -3.1329e-02 | 0/4 | 1/4 | +5.9218e-02 | +0.0000e+00 | +3.2022e-02 | -4.1322e-03 |
| DBH76 | 70 | 0.9310 | 0.2596 | -2.2432e-02 | 0.9574 | 0.2670 | -1.5069e-02 | +7.3633e-03 | 65/70 | 64/70 | +6.2233e-04 | -2.3055e-02 | +1.1770e-03 | -1.6246e-02 |
| EA13 | 11 | 1.8706 | 0.0820 | +6.8503e-03 | 1.8132 | 0.0795 | +4.3350e-03 | -2.5153e-03 | 5/11 | 4/11 | +2.3146e-02 | -1.6296e-02 | +1.6056e-02 | -1.1721e-02 |
| IP13 | 13 | 2.5806 | 0.1337 | +1.1367e-02 | 2.4146 | 0.1251 | +2.7724e-03 | -8.5945e-03 | 7/13 | 4/13 | +1.7342e-02 | -5.9749e-03 | +8.3200e-03 | -5.5476e-03 |
| MGAE109 | 104 | 0.2078 | 0.0861 | -3.0744e-02 | 0.2552 | 0.1057 | -1.1117e-02 | +1.9627e-02 | 91/104 | 79/104 | +2.2678e-03 | -3.3012e-02 | +1.8258e-03 | -1.2943e-02 |
| NCCE31 | 28 | 2.1041 | 0.2347 | -5.7285e-02 | 2.2429 | 0.2502 | -4.1790e-02 | +1.5494e-02 | 26/28 | 27/28 | +2.8063e-03 | -6.0091e-02 | +4.1890e-06 | -4.1795e-02 |
| PA8 | 8 | 2.5529 | 0.0814 | +2.9545e-02 | 2.2812 | 0.0727 | +2.0885e-02 | -8.6599e-03 | 1/8 | 2/8 | +3.7168e-02 | -7.6235e-03 | +3.0779e-02 | -9.8944e-03 |
| pTC13 | 13 | 5.1068 | 0.2645 | +5.8593e-02 | 4.9535 | 0.2566 | +5.0653e-02 | -7.9403e-03 | 3/13 | 2/13 | +6.1138e-02 | -2.5454e-03 | +5.0738e-02 | -8.5469e-05 |

Sign of the database delta across every audited checkpoint:

- ABDE4 worsens at every audited checkpoint
- DBH76 improves at every audited checkpoint
- MGAE109 improves at every audited checkpoint
- NCCE31 improves at every audited checkpoint
- PA8 worsens at every audited checkpoint
- pTC13 worsens at every audited checkpoint

### How concentrated the worsening is

| Checkpoint | Top 5 / positive mass | Top 10 / positive mass | Top 20 / positive mass | Top 20 / net rise |
|---|---:|---:|---:|---:|
| iid_t59 | 0.407 | 0.561 | 0.803 | 6.591 |
| iid_t70 | 0.365 | 0.532 | 0.793 | 2.930 |
| iid_t80 | 0.385 | 0.544 | 0.801 | 6.142 |
| iid_t90 | 0.318 | 0.518 | 0.800 | 2.988 |
| dbstrat_t70 | 0.346 | 0.556 | 0.842 | 3.077 |
| dbstrat_t90 | 0.250 | 0.437 | 0.785 | 2.965 |
| lr3e-5_t80 | 0.372 | 0.538 | 0.796 | 3.626 |

### Twenty largest worsenings at iid_t70

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_499996e5b8084d7129c856a5` | ABDE4 | level3 | 11.499434 | 16.903200 | +5.403767 |
| `reaction_c258198ad576955c8267a32b` | ABDE4 | level3_delley | 2.482637 | 7.819478 | +5.336841 |
| `reaction_18a4cfbaf87fde8157323b2d` | ABDE4 | level3_mura | 6.841078 | 10.432474 | +3.591396 |
| `reaction_dce81e2069e365d6f6f375e6` | PA8 | level2_delley | 3.303584 | 5.905796 | +2.602212 |
| `reaction_2711e62a8820baac39b75e55` | pTC13 | level3 | 4.289338 | 6.037538 | +1.748200 |
| `reaction_976cf34c643b73837e9db4ea` | pTC13 | level2_delley | 3.264877 | 5.009067 | +1.744191 |
| `reaction_1713f94bc8687f01854a6033` | pTC13 | level3 | 5.698730 | 7.440680 | +1.741950 |
| `reaction_a3c28ec21592df8905d784c1` | pTC13 | level3_gauss_chebyshev | 0.333040 | 2.063471 | +1.730431 |
| `reaction_98e0120f8190c1b6ea1d880f` | pTC13 | level2_mura | 2.085056 | 3.810007 | +1.724951 |
| `reaction_22b428ed93dc6e716de3a64c` | PA8 | level3_delley | 1.334741 | 2.928513 | +1.593772 |
| `reaction_0fd412eae12856c37024d4c7` | EA13 | level2_delley | 1.142234 | 2.723827 | +1.581592 |
| `reaction_48423cf9420faf0d8c8583ec` | PA8 | level2_gauss_chebyshev | 2.869198 | 4.431963 | +1.562766 |
| `reaction_ab43d4186b8395ac22ace39a` | pTC13 | level3_delley | 5.298021 | 6.702877 | +1.404855 |
| `reaction_d880804b7d24db2928691e9a` | pTC13 | level2_mura | 1.800608 | 3.205177 | +1.404569 |
| `reaction_e5643cb85ab3252226463941` | pTC13 | level3_gauss_chebyshev | 3.640986 | 5.041715 | +1.400728 |
| `reaction_9d13c140c1b6ecee985a835c` | pTC13 | level3 | 7.095213 | 8.492827 | +1.397614 |
| `reaction_a033d2adfacdefc08b74793b` | PA8 | level3 | 0.379594 | 1.632963 | +1.253369 |
| `reaction_a9a0d336c0d974685834be56` | PA8 | level2_delley | 1.168190 | 2.411267 | +1.243076 |
| `reaction_32ca1bae32af42e9dfcca203` | pTC13 | level2_mura | 0.153736 | 1.201974 | +1.048238 |
| `reaction_6e10bb0848f1487cc56a51ca` | IP13 | level3 | 0.662307 | 1.676812 | +1.014505 |

### Twenty largest improvements at iid_t70

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_b77dc0163ad1d8e3ce64f6ec` | NCCE31 | level2_mura | 3.899487 | 1.822939 | -2.076547 |
| `reaction_2141fbc9d785ade28191a592` | PA8 | level3_gauss_chebyshev | 2.463234 | 0.549732 | -1.913502 |
| `reaction_3d712f23813911dbbfdce318` | EA13 | level2_mura | 2.130007 | 0.479578 | -1.650428 |
| `reaction_fb9cfb97266cee37ae5c9a79` | NCCE31 | level2_delley | 1.946939 | 0.422877 | -1.524062 |
| `reaction_26d720b15604b2c1a8506792` | NCCE31 | level2_delley | 8.963101 | 7.684784 | -1.278317 |
| `reaction_a86e803a19d9560fc35c4035` | NCCE31 | level3_gauss_chebyshev | 8.392713 | 7.138407 | -1.254306 |
| `reaction_cb3e3d8f936d74fed1dd9c3b` | NCCE31 | level2 | 5.383748 | 4.217064 | -1.166684 |
| `reaction_082486d89aa465ca2fc8e2a9` | NCCE31 | level3 | 3.539089 | 2.395283 | -1.143806 |
| `reaction_97e9af9791a68e72d7655cca` | EA13 | level3_delley | 4.832930 | 3.807005 | -1.025925 |
| `reaction_343822c882026576bd14704a` | EA13 | level3_mura | 1.235968 | 0.438859 | -0.797109 |
| `reaction_413ff08c7b15bcb55c063d73` | NCCE31 | level3_delley | 2.606341 | 1.855057 | -0.751285 |
| `reaction_a932b7106b3ffa0001c2236d` | NCCE31 | level2 | 1.018131 | 0.368166 | -0.649965 |
| `reaction_4a9cc9bd07a5b2c53beb7427` | NCCE31 | level3_gauss_chebyshev | 6.739883 | 6.090542 | -0.649342 |
| `reaction_1a6059b2409d5d08512f805c` | NCCE31 | level2 | 0.784303 | 0.207643 | -0.576660 |
| `reaction_2304bad71e7d0e1921147c13` | IP13 | level3_mura | 7.021024 | 6.503342 | -0.517682 |
| `reaction_68864e56d1890c516c29f333` | NCCE31 | level2_gauss_chebyshev | 1.731373 | 1.218596 | -0.512777 |
| `reaction_9bb40e5f412a46fa89c24027` | NCCE31 | level3_mura | 3.735067 | 3.291261 | -0.443806 |
| `reaction_cf4db7a9604b3a66bd4ab78b` | NCCE31 | level2 | 7.481627 | 7.052144 | -0.429484 |
| `reaction_09347705da4214adc37a6cfd` | NCCE31 | level2_mura | 2.356770 | 1.961724 | -0.395046 |
| `reaction_3955b294f22cd0d31f796834` | NCCE31 | level3_delley | 1.416650 | 1.029444 | -0.387206 |

### Twenty largest worsenings at dbstrat_t70

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_499996e5b8084d7129c856a5` | ABDE4 | level3 | 11.499434 | 15.098840 | +3.599406 |
| `reaction_c258198ad576955c8267a32b` | ABDE4 | level3_delley | 2.482637 | 6.080789 | +3.598152 |
| `reaction_dce81e2069e365d6f6f375e6` | PA8 | level2_delley | 3.303584 | 5.159029 | +1.855444 |
| `reaction_22b428ed93dc6e716de3a64c` | PA8 | level3_delley | 1.334741 | 2.992047 | +1.657307 |
| `reaction_2711e62a8820baac39b75e55` | pTC13 | level3 | 4.289338 | 5.822658 | +1.533319 |
| `reaction_1713f94bc8687f01854a6033` | pTC13 | level3 | 5.698730 | 7.231899 | +1.533170 |
| `reaction_976cf34c643b73837e9db4ea` | pTC13 | level2_delley | 3.264877 | 4.795004 | +1.530128 |
| `reaction_98e0120f8190c1b6ea1d880f` | pTC13 | level2_mura | 2.085056 | 3.582827 | +1.497771 |
| `reaction_48423cf9420faf0d8c8583ec` | PA8 | level2_gauss_chebyshev | 2.869198 | 4.356242 | +1.487044 |
| `reaction_a3c28ec21592df8905d784c1` | pTC13 | level3_gauss_chebyshev | 0.333040 | 1.691035 | +1.357995 |
| `reaction_a033d2adfacdefc08b74793b` | PA8 | level3 | 0.379594 | 1.574909 | +1.195314 |
| `reaction_a9a0d336c0d974685834be56` | PA8 | level2_delley | 1.168190 | 2.346474 | +1.178284 |
| `reaction_e5643cb85ab3252226463941` | pTC13 | level3_gauss_chebyshev | 3.640986 | 4.773402 | +1.132416 |
| `reaction_ab43d4186b8395ac22ace39a` | pTC13 | level3_delley | 5.298021 | 6.421097 | +1.123076 |
| `reaction_d880804b7d24db2928691e9a` | pTC13 | level2_mura | 1.800608 | 2.923573 | +1.122966 |
| `reaction_9d13c140c1b6ecee985a835c` | pTC13 | level3 | 7.095213 | 8.198936 | +1.103723 |
| `reaction_0fd412eae12856c37024d4c7` | EA13 | level2_delley | 1.142234 | 2.024168 | +0.881934 |
| `reaction_18a4cfbaf87fde8157323b2d` | ABDE4 | level3_mura | 6.841078 | 7.681024 | +0.839946 |
| `reaction_32ca1bae32af42e9dfcca203` | pTC13 | level2_mura | 0.153736 | 0.932739 | +0.779003 |
| `reaction_e2265998297f22c00106c8d5` | EA13 | level3_delley | 0.144582 | 0.913859 | +0.769277 |

### Twenty largest improvements at dbstrat_t70

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_2141fbc9d785ade28191a592` | PA8 | level3_gauss_chebyshev | 2.463234 | 0.422439 | -2.040795 |
| `reaction_b77dc0163ad1d8e3ce64f6ec` | NCCE31 | level2_mura | 3.899487 | 2.078246 | -1.821241 |
| `reaction_3d712f23813911dbbfdce318` | EA13 | level2_mura | 2.130007 | 0.895402 | -1.234605 |
| `reaction_fb9cfb97266cee37ae5c9a79` | NCCE31 | level2_delley | 1.946939 | 0.849920 | -1.097019 |
| `reaction_99331671b618716acb0d2398` | ABDE4 | level2_mura | 1.618673 | 0.581493 | -1.037180 |
| `reaction_26d720b15604b2c1a8506792` | NCCE31 | level2_delley | 8.963101 | 8.183631 | -0.779469 |
| `reaction_082486d89aa465ca2fc8e2a9` | NCCE31 | level3 | 3.539089 | 2.764714 | -0.774376 |
| `reaction_a86e803a19d9560fc35c4035` | NCCE31 | level3_gauss_chebyshev | 8.392713 | 7.629346 | -0.763367 |
| `reaction_cb3e3d8f936d74fed1dd9c3b` | NCCE31 | level2 | 5.383748 | 4.636460 | -0.747288 |
| `reaction_7e17106b627c62588aff9af0` | EA13 | level2_delley | 1.586709 | 0.909303 | -0.677406 |
| `reaction_dc222cca36aa35ad81344f36` | IP13 | level3_mura | 1.678180 | 1.010160 | -0.668020 |
| `reaction_413ff08c7b15bcb55c063d73` | NCCE31 | level3_delley | 2.606341 | 2.025989 | -0.580352 |
| `reaction_343822c882026576bd14704a` | EA13 | level3_mura | 1.235968 | 0.658642 | -0.577326 |
| `reaction_779c61a8f1686c7811b5f243` | IP13 | level2_gauss_chebyshev | 0.651905 | 0.105323 | -0.546583 |
| `reaction_a932b7106b3ffa0001c2236d` | NCCE31 | level2 | 1.018131 | 0.500301 | -0.517830 |
| `reaction_97e9af9791a68e72d7655cca` | EA13 | level3_delley | 4.832930 | 4.380229 | -0.452700 |
| `reaction_bc6539fdc862d79194cf76dc` | PA8 | level3_gauss_chebyshev | 0.756451 | 0.313747 | -0.442704 |
| `reaction_1a6059b2409d5d08512f805c` | NCCE31 | level2 | 0.784303 | 0.354123 | -0.430181 |
| `reaction_4a9cc9bd07a5b2c53beb7427` | NCCE31 | level3_gauss_chebyshev | 6.739883 | 6.311966 | -0.427917 |
| `reaction_68864e56d1890c516c29f333` | NCCE31 | level2_gauss_chebyshev | 1.731373 | 1.328707 | -0.402666 |

### Twenty largest worsenings at iid_t90

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_18a4cfbaf87fde8157323b2d` | ABDE4 | level3_mura | 6.841078 | 9.242771 | +2.401694 |
| `reaction_499996e5b8084d7129c856a5` | ABDE4 | level3 | 11.499434 | 13.845240 | +2.345806 |
| `reaction_c258198ad576955c8267a32b` | ABDE4 | level3_delley | 2.482637 | 4.590737 | +2.108100 |
| `reaction_5b01706a99497ca30662a17d` | IP13 | level2_mura | 4.237946 | 5.695083 | +1.457138 |
| `reaction_dce81e2069e365d6f6f375e6` | PA8 | level2_delley | 3.303584 | 4.667515 | +1.363931 |
| `reaction_fcbc6f8c1a9c1f8c5b31b1ac` | IP13 | level2_gauss_chebyshev | 1.395225 | 2.690231 | +1.295006 |
| `reaction_62fe6712dbf4bdeda18637d6` | IP13 | level2_delley | 0.705412 | 1.982476 | +1.277063 |
| `reaction_dc222cca36aa35ad81344f36` | IP13 | level3_mura | 1.678180 | 2.943089 | +1.264910 |
| `reaction_6e10bb0848f1487cc56a51ca` | IP13 | level3 | 0.662307 | 1.829341 | +1.167034 |
| `reaction_c951e12704d4e9f0d83d7faa` | IP13 | level3_mura | 2.529500 | 3.606900 | +1.077400 |
| `reaction_9d13c140c1b6ecee985a835c` | pTC13 | level3 | 7.095213 | 8.040048 | +0.944835 |
| `reaction_ab43d4186b8395ac22ace39a` | pTC13 | level3_delley | 5.298021 | 6.233063 | +0.935041 |
| `reaction_e5643cb85ab3252226463941` | pTC13 | level3_gauss_chebyshev | 3.640986 | 4.560914 | +0.919927 |
| `reaction_d880804b7d24db2928691e9a` | pTC13 | level2_mura | 1.800608 | 2.701209 | +0.900601 |
| `reaction_a3c28ec21592df8905d784c1` | pTC13 | level3_gauss_chebyshev | 0.333040 | 1.170469 | +0.837429 |
| `reaction_1713f94bc8687f01854a6033` | pTC13 | level3 | 5.698730 | 6.531120 | +0.832390 |
| `reaction_2711e62a8820baac39b75e55` | pTC13 | level3 | 4.289338 | 5.109754 | +0.820416 |
| `reaction_22b428ed93dc6e716de3a64c` | PA8 | level3_delley | 1.334741 | 2.148247 | +0.813507 |
| `reaction_976cf34c643b73837e9db4ea` | pTC13 | level2_delley | 3.264877 | 4.065699 | +0.800823 |
| `reaction_98e0120f8190c1b6ea1d880f` | pTC13 | level2_mura | 2.085056 | 2.864796 | +0.779740 |

### Twenty largest improvements at iid_t90

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_2141fbc9d785ade28191a592` | PA8 | level3_gauss_chebyshev | 2.463234 | 1.418934 | -1.044300 |
| `reaction_99331671b618716acb0d2398` | ABDE4 | level2_mura | 1.618673 | 0.609235 | -1.009438 |
| `reaction_a86e803a19d9560fc35c4035` | NCCE31 | level3_gauss_chebyshev | 8.392713 | 7.519526 | -0.873187 |
| `reaction_26d720b15604b2c1a8506792` | NCCE31 | level2_delley | 8.963101 | 8.103491 | -0.859609 |
| `reaction_cb3e3d8f936d74fed1dd9c3b` | NCCE31 | level2 | 5.383748 | 4.632058 | -0.751690 |
| `reaction_d6710986efd6e92ceab03093` | IP13 | level3_delley | 6.122183 | 5.404706 | -0.717476 |
| `reaction_082486d89aa465ca2fc8e2a9` | NCCE31 | level3 | 3.539089 | 2.841791 | -0.697298 |
| `reaction_fb9cfb97266cee37ae5c9a79` | NCCE31 | level2_delley | 1.946939 | 1.318456 | -0.628483 |
| `reaction_25eb7859a60d72318d2d165c` | EA13 | level2_gauss_chebyshev | 0.688206 | 0.060638 | -0.627568 |
| `reaction_b77dc0163ad1d8e3ce64f6ec` | NCCE31 | level2_mura | 3.899487 | 3.303806 | -0.595681 |
| `reaction_4a9cc9bd07a5b2c53beb7427` | NCCE31 | level3_gauss_chebyshev | 6.739883 | 6.360400 | -0.379483 |
| `reaction_68864e56d1890c516c29f333` | NCCE31 | level2_gauss_chebyshev | 1.731373 | 1.364370 | -0.367003 |
| `reaction_cf4db7a9604b3a66bd4ab78b` | NCCE31 | level2 | 7.481627 | 7.134762 | -0.346865 |
| `reaction_bc6539fdc862d79194cf76dc` | PA8 | level3_gauss_chebyshev | 0.756451 | 0.418775 | -0.337676 |
| `reaction_413ff08c7b15bcb55c063d73` | NCCE31 | level3_delley | 2.606341 | 2.294048 | -0.312293 |
| `reaction_3d712f23813911dbbfdce318` | EA13 | level2_mura | 2.130007 | 1.846942 | -0.283064 |
| `reaction_cf9ba2c7d1ce7be6b152f41a` | NCCE31 | level2 | 0.424818 | 0.154362 | -0.270455 |
| `reaction_9bb40e5f412a46fa89c24027` | NCCE31 | level3_mura | 3.735067 | 3.469989 | -0.265078 |
| `reaction_a932b7106b3ffa0001c2236d` | NCCE31 | level2 | 1.018131 | 0.770562 | -0.247569 |
| `reaction_09347705da4214adc37a6cfd` | NCCE31 | level2_mura | 2.356770 | 2.111415 | -0.245355 |

### Twenty largest worsenings at dbstrat_t90

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_499996e5b8084d7129c856a5` | ABDE4 | level3 | 11.499434 | 12.879931 | +1.380498 |
| `reaction_c258198ad576955c8267a32b` | ABDE4 | level3_delley | 2.482637 | 3.687776 | +1.205139 |
| `reaction_dce81e2069e365d6f6f375e6` | PA8 | level2_delley | 3.303584 | 4.426533 | +1.122949 |
| `reaction_22b428ed93dc6e716de3a64c` | PA8 | level3_delley | 1.334741 | 2.289993 | +0.955253 |
| `reaction_5b01706a99497ca30662a17d` | IP13 | level2_mura | 4.237946 | 5.093616 | +0.855671 |
| `reaction_1713f94bc8687f01854a6033` | pTC13 | level3 | 5.698730 | 6.540853 | +0.842124 |
| `reaction_2711e62a8820baac39b75e55` | pTC13 | level3 | 4.289338 | 5.124958 | +0.835620 |
| `reaction_976cf34c643b73837e9db4ea` | pTC13 | level2_delley | 3.264877 | 4.091685 | +0.826809 |
| `reaction_62fe6712dbf4bdeda18637d6` | IP13 | level2_delley | 0.705412 | 1.528850 | +0.823438 |
| `reaction_e5643cb85ab3252226463941` | pTC13 | level3_gauss_chebyshev | 3.640986 | 4.457023 | +0.816037 |
| `reaction_ab43d4186b8395ac22ace39a` | pTC13 | level3_delley | 5.298021 | 6.112895 | +0.814873 |
| `reaction_48423cf9420faf0d8c8583ec` | PA8 | level2_gauss_chebyshev | 2.869198 | 3.680614 | +0.811416 |
| `reaction_9d13c140c1b6ecee985a835c` | pTC13 | level3 | 7.095213 | 7.903658 | +0.808445 |
| `reaction_98e0120f8190c1b6ea1d880f` | pTC13 | level2_mura | 2.085056 | 2.889300 | +0.804245 |
| `reaction_d880804b7d24db2928691e9a` | pTC13 | level2_mura | 1.800608 | 2.598176 | +0.797568 |
| `reaction_fcbc6f8c1a9c1f8c5b31b1ac` | IP13 | level2_gauss_chebyshev | 1.395225 | 2.167632 | +0.772407 |
| `reaction_a3c28ec21592df8905d784c1` | pTC13 | level3_gauss_chebyshev | 0.333040 | 1.099255 | +0.766215 |
| `reaction_6e10bb0848f1487cc56a51ca` | IP13 | level3 | 0.662307 | 1.401511 | +0.739204 |
| `reaction_a9a0d336c0d974685834be56` | PA8 | level2_delley | 1.168190 | 1.886029 | +0.717839 |
| `reaction_c951e12704d4e9f0d83d7faa` | IP13 | level3_mura | 2.529500 | 3.193219 | +0.663719 |

### Twenty largest improvements at dbstrat_t90

| Identity | Database | Variant | P536 loss | Checkpoint loss | Difference |
|---|---|---|---:|---:|---:|
| `reaction_2141fbc9d785ade28191a592` | PA8 | level3_gauss_chebyshev | 2.463234 | 1.219655 | -1.243579 |
| `reaction_b77dc0163ad1d8e3ce64f6ec` | NCCE31 | level2_mura | 3.899487 | 3.124922 | -0.774565 |
| `reaction_a86e803a19d9560fc35c4035` | NCCE31 | level3_gauss_chebyshev | 8.392713 | 7.726606 | -0.666107 |
| `reaction_bc6539fdc862d79194cf76dc` | PA8 | level3_gauss_chebyshev | 0.756451 | 0.093096 | -0.663355 |
| `reaction_26d720b15604b2c1a8506792` | NCCE31 | level2_delley | 8.963101 | 8.320200 | -0.642901 |
| `reaction_cb3e3d8f936d74fed1dd9c3b` | NCCE31 | level2 | 5.383748 | 4.786255 | -0.597494 |
| `reaction_082486d89aa465ca2fc8e2a9` | NCCE31 | level3 | 3.539089 | 2.955954 | -0.583135 |
| `reaction_fb9cfb97266cee37ae5c9a79` | NCCE31 | level2_delley | 1.946939 | 1.366012 | -0.580927 |
| `reaction_d6710986efd6e92ceab03093` | IP13 | level3_delley | 6.122183 | 5.550950 | -0.571233 |
| `reaction_779c61a8f1686c7811b5f243` | IP13 | level2_gauss_chebyshev | 0.651905 | 0.268077 | -0.383828 |
| `reaction_25eb7859a60d72318d2d165c` | EA13 | level2_gauss_chebyshev | 0.688206 | 0.344381 | -0.343825 |
| `reaction_68864e56d1890c516c29f333` | NCCE31 | level2_gauss_chebyshev | 1.731373 | 1.398644 | -0.332729 |
| `reaction_cf9ba2c7d1ce7be6b152f41a` | NCCE31 | level2 | 0.424818 | 0.108034 | -0.316783 |
| `reaction_413ff08c7b15bcb55c063d73` | NCCE31 | level3_delley | 2.606341 | 2.299807 | -0.306534 |
| `reaction_dedcd9337297c7d45c65bc9a` | PA8 | level2_gauss_chebyshev | 0.732287 | 0.434144 | -0.298142 |
| `reaction_ff7f9ec365d4032a12fe400a` | NCCE31 | level2_gauss_chebyshev | 1.429844 | 1.139559 | -0.290285 |
| `reaction_4a9cc9bd07a5b2c53beb7427` | NCCE31 | level3_gauss_chebyshev | 6.739883 | 6.451296 | -0.288587 |
| `reaction_99331671b618716acb0d2398` | ABDE4 | level2_mura | 1.618673 | 1.341190 | -0.277484 |
| `reaction_cf4db7a9604b3a66bd4ab78b` | NCCE31 | level2 | 7.481627 | 7.205939 | -0.275688 |
| `reaction_3d712f23813911dbbfdce318` | EA13 | level2_mura | 2.130007 | 1.870797 | -0.259210 |

## IID versus database-stratified relchem

At t70, 43 identities are worse than P536 in both streams, and 10 are worse under IID but improved under DB-strat.
At t90 those counts are 42 and 7.

### Largest reactions worse in both streams at t70

| Identity | Database | Variant | P536 loss | IID loss | DB-strat loss | IID difference | DB-strat difference |
|---|---|---|---:|---:|---:|---:|---:|
| `reaction_499996e5b8084d7129c856a5` | ABDE4 | level3 | 11.499434 | 16.903200 | 15.098840 | +5.403767 | +3.599406 |
| `reaction_c258198ad576955c8267a32b` | ABDE4 | level3_delley | 2.482637 | 7.819478 | 6.080789 | +5.336841 | +3.598152 |
| `reaction_dce81e2069e365d6f6f375e6` | PA8 | level2_delley | 3.303584 | 5.905796 | 5.159029 | +2.602212 | +1.855444 |
| `reaction_18a4cfbaf87fde8157323b2d` | ABDE4 | level3_mura | 6.841078 | 10.432474 | 7.681024 | +3.591396 | +0.839946 |
| `reaction_2711e62a8820baac39b75e55` | pTC13 | level3 | 4.289338 | 6.037538 | 5.822658 | +1.748200 | +1.533319 |
| `reaction_1713f94bc8687f01854a6033` | pTC13 | level3 | 5.698730 | 7.440680 | 7.231899 | +1.741950 | +1.533170 |
| `reaction_976cf34c643b73837e9db4ea` | pTC13 | level2_delley | 3.264877 | 5.009067 | 4.795004 | +1.744191 | +1.530128 |
| `reaction_22b428ed93dc6e716de3a64c` | PA8 | level3_delley | 1.334741 | 2.928513 | 2.992047 | +1.593772 | +1.657307 |
| `reaction_98e0120f8190c1b6ea1d880f` | pTC13 | level2_mura | 2.085056 | 3.810007 | 3.582827 | +1.724951 | +1.497771 |
| `reaction_a3c28ec21592df8905d784c1` | pTC13 | level3_gauss_chebyshev | 0.333040 | 2.063471 | 1.691035 | +1.730431 | +1.357995 |
| `reaction_48423cf9420faf0d8c8583ec` | PA8 | level2_gauss_chebyshev | 2.869198 | 4.431963 | 4.356242 | +1.562766 | +1.487044 |
| `reaction_e5643cb85ab3252226463941` | pTC13 | level3_gauss_chebyshev | 3.640986 | 5.041715 | 4.773402 | +1.400728 | +1.132416 |
| `reaction_ab43d4186b8395ac22ace39a` | pTC13 | level3_delley | 5.298021 | 6.702877 | 6.421097 | +1.404855 | +1.123076 |
| `reaction_d880804b7d24db2928691e9a` | pTC13 | level2_mura | 1.800608 | 3.205177 | 2.923573 | +1.404569 | +1.122966 |
| `reaction_9d13c140c1b6ecee985a835c` | pTC13 | level3 | 7.095213 | 8.492827 | 8.198936 | +1.397614 | +1.103723 |
| `reaction_0fd412eae12856c37024d4c7` | EA13 | level2_delley | 1.142234 | 2.723827 | 2.024168 | +1.581592 | +0.881934 |
| `reaction_a033d2adfacdefc08b74793b` | PA8 | level3 | 0.379594 | 1.632963 | 1.574909 | +1.253369 | +1.195314 |
| `reaction_a9a0d336c0d974685834be56` | PA8 | level2_delley | 1.168190 | 2.411267 | 2.346474 | +1.243076 | +1.178284 |
| `reaction_32ca1bae32af42e9dfcca203` | pTC13 | level2_mura | 0.153736 | 1.201974 | 0.932739 | +1.048238 | +0.779003 |
| `reaction_e2265998297f22c00106c8d5` | EA13 | level3_delley | 0.144582 | 1.107753 | 0.913859 | +0.963171 | +0.769277 |

### Worse under IID t70 and improved under DB-strat t70

| Identity | Database | Variant | P536 loss | IID loss | DB-strat loss | IID difference | DB-strat difference |
|---|---|---|---:|---:|---:|---:|---:|
| `reaction_99331671b618716acb0d2398` | ABDE4 | level2_mura | 1.618673 | 2.150478 | 0.581493 | +0.531805 | -1.037180 |
| `reaction_bc6539fdc862d79194cf76dc` | PA8 | level3_gauss_chebyshev | 0.756451 | 1.025136 | 0.313747 | +0.268685 | -0.442704 |
| `reaction_ff7f9ec365d4032a12fe400a` | NCCE31 | level2_gauss_chebyshev | 1.429844 | 1.787965 | 1.399629 | +0.358121 | -0.030215 |
| `reaction_ac624171c264f80a39188188` | MGAE109 | level3_delley | 0.208892 | 0.245556 | 0.193774 | +0.036664 | -0.015117 |
| `reaction_24b475b5fe1fff22333ade23` | DBH76 | level2_delley | 0.701382 | 0.726183 | 0.694614 | +0.024802 | -0.006768 |
| `reaction_7b80bb94ea04577d804d48a5` | MGAE109 | level3_gauss_chebyshev | 0.173110 | 0.192411 | 0.163860 | +0.019301 | -0.009250 |
| `reaction_aeb891a96c75e5959b50027a` | DBH76 | level3_mura | 0.705480 | 0.724474 | 0.698288 | +0.018994 | -0.007192 |
| `reaction_307e28560e02ab87bb3f0c03` | DBH76 | level3_delley | 0.705496 | 0.724494 | 0.698311 | +0.018998 | -0.007185 |
| `reaction_785059796d82f49b53ca3870` | MGAE109 | level2_mura | 0.079071 | 0.086180 | 0.069318 | +0.007109 | -0.009753 |
| `reaction_aad962147a2f6c2276c05ed5` | MGAE109 | level2_delley | 0.080745 | 0.087310 | 0.076270 | +0.006565 | -0.004475 |

## Sampling exposure

Counts below are the 90 executed relchem draws. They are not claims about the earlier P536 predopt phase.

| Database | IID samples | IID unique | DB-strat samples | DB-strat unique |
|---|---:|---:|---:|---:|
| ABDE4 | 0 | 0 | 8 | 4 |
| DBH76 | 24 | 20 | 6 | 6 |
| EA13 | 5 | 5 | 18 | 9 |
| IP13 | 4 | 3 | 9 | 6 |
| MGAE109 | 39 | 34 | 11 | 10 |
| NCCE31 | 9 | 8 | 14 | 11 |
| PA8 | 5 | 4 | 9 | 7 |
| pTC13 | 4 | 4 | 15 | 8 |

Mean qualified-loss change at t70 by whether the identity appeared in that arm's 90 draws:

- iid_t70: unseen_in_90 n=173 mean delta=+0.1046, seen_once n=66 mean delta=-0.0601, seen_more_than_once n=12 mean delta=-0.0254
- dbstrat_t70: unseen_in_90 n=190 mean delta=-0.0286, seen_once n=41 mean delta=+0.0934, seen_more_than_once n=20 mean delta=+0.5643

Median logged raw relchem gradient norm over 90 updates: IID 13.803; DB-strat raw 166.995; DB-strat after multiplying by the importance weight once 66.752. The IID median matches the paired control series already stored in the DB-strat metrics. The larger DB-strat raw norms follow the databases that sampler draws: ABDE4, PA8, and pTC13 have much larger raw norms than MGAE109 and DBH76.

Of the 20 largest IID t70 worsenings, 18 identities are absent from the IID manifest and 1 include the frozen evaluation variant. On the DB-strat manifest, 7 of those identities are absent and 2 include the frozen variant. The fixed-panel losses therefore moved without repeated training on the evaluated grid. Under uniform identity sampling, the probability of drawing no ABDE4 identity in 90 updates is 0.236. That makes one empty ABDE4 stream plausible. It does not show that those identities were absent from the earlier P536 predopt phase.

The exposure bins are descriptive. In the IID stream, unseen identities account for the net rise and seen identities improve on average. In the DB-strat stream the repeated identities are the small databases, and those repeats have a positive mean change. That association is confounded by which databases receive the repeats. It is not a causal estimate of exposure.

## Clean28

Clean28 is the mean Diet-weighted absolute error of 28 leakage-clean reactions. It is not the 251-identity relchem objective. Identifiers differ (`ACONF-10` versus `reaction_<hash>`), so the panels were not joined.

| Checkpoint | Clean28 |
|---|---:|
| t0 | 9.553190636 |
| iid_t70 | 8.629660230 |
| dbstrat_t70 | 8.909938464 |
| lr3e-5_t80 | 8.619694172 |

Clean28 falls at every cited checkpoint while the 251-identity relchem mean rises, so Clean28 is not a proxy for relchem eligibility. Diet30 subset labels on that external panel are stored in the metrics. They are not the eight training databases, and similarly named reactions were not treated as the same identity.

## Skala-1.1 v6, from arXiv:2506.14665v6

The current manuscript identifies Skala-1.1, trained on about 400,000 energy differences, as superseding the earlier Skala-1.0 recipe. Facts below are from the fetched v6 text.

Pretraining evaluates the functional on fixed B3LYP densities and precomputed non-XC total-energy components, including D3 (main text Sec. 2.1 and Supplement B.1, Eq. 29). The loss is the expectation of squared reaction-energy error divided by `1e-4 Eh + |reference reaction energy|` (B.1, Eq. 31). That is not the qualified singleton loss used here, which is a database-and-frequency-weighted square root of a one-reaction MSE.

Sampling is two-level: a dataset is drawn with probability `p_i`, then a reaction is drawn uniformly inside it (B.2). Initial `p_i` is proportional to category weight times dataset size. Nine categories are named. The numeric category weights and target proportions are NOT VERIFIED; the extracted B.2 says they are solved to hit prescribed proportions but does not list them. Every 25,000 steps, relative excess loss against baseline DFT MAEs multiplies those probabilities by `exp(0.025 * REL)` and renormalizes them (B.2, Eqs. 32-33). This is not uniform-database sampling followed by an inverse-probability correction. The B.2 title includes model selection. An explicit checkpoint-picking rule beyond the REL diagnostic was NOT VERIFIED in the extracted section.

Optimization uses Muon on hidden matrices and Adam on biases and the final layer, separate cosine schedules with 50,000 warmup steps, 1,000,000 pretraining steps, a gradient-clipping threshold of 0.0001, and EMA decay 0.9999 (B.4, Eq. 34 and Table 2). Peak learning rates are 0.0007 for Muon and 0.00015 for Adam. The numeric cosine floor is NOT VERIFIED. Ablations used 8 A100 GPUs with one reaction per GPU. This is not constant-LR AdamW for 90 updates.

Fine-tuning runs 20,000 steps on the model's own SCF densities, with the optimizer reset, constant learning rate 1e-5, the same weighted loss, and sampling probabilities frozen from step 1,000,000 (B.5). No gradient is propagated through SCF. That procedure is not authorized here and was not run.

| Component | Skala-1.1 v6 | Current piNN-DFT 90-update arms | Potential relevance |
|---|---|---|---|
| Density source | Fixed B3LYP densities, not B3LYP energies, in pretraining; the model's own SCF densities in fine-tuning (Sec. 2.1, B.1, B.5) | Frozen Minnesota reaction grids for the 251-identity objective. Clean28 uses frozen PBE0 densities and PBE0-D3(BJ). The SCF functional that generated the Minnesota grids is NOT VERIFIED here | Both pretraining stages evaluate a fixed density. The density sources are not the same |
| Reaction regression loss | Weighted MSE, Eq. 31 | Qualified weighted sqrt-MSE singleton, then an equal-identity mean | The objective being audited is not Skala's reaction MSE |
| Sampling probabilities | Hierarchical, then REL-adaptive | IID uniform identity, or uniform database plus `p/q` | Skala's adaptation changes effort toward weak datasets; our correction preserves the IID singleton expectation |
| Chemical dataset balance | Category targets, then excess-loss updates | Frozen Minnesota database/frequency factors inside the singleton loss | Our factors are constant weights, not an online sampler |
| Optimizer | Muon plus Adam, warmup and cosine | Constant AdamW, lr 1e-4 or one reduced-lr branch | No evidence yet that the optimizer family explains the relchem ratio |
| Training duration | 1,000,000 pretraining steps plus 20,000 fine-tuning steps | 536 PBE predopt updates, then 90 AdamW updates | The audited failure is inside the short AdamW phase |
| Model stabilization | EMA 0.9999, gradient clipping | No EMA and no clipping change in these arms | NOT VERIFIED as a cause of the relchem ratio |
| SCF fine-tuning | 20,000 on-policy SCF steps | Not authorized | Cannot explain these fixed-density relchem numbers |
| Scientific constraints | Enhancement-factor constraints including uniform scaling, size consistency, and a Lieb-Oxford bound | Tau-free Laplacian NN-PBE with the existing PBE constraints | Both constrain the XC form; the constraint sets are not the same |

Skala demonstrates a long hierarchical pretraining run and a separate SCF fine-tune on a much larger reaction collection. Our receipts demonstrate that 90 AdamW updates improve AE17, Exc, and the operator while the fixed 251-identity relchem mean rises. The comparison suggests hypotheses. It does not show that copying Skala's sampler would lower this relchem objective.

## Next experiment, not executed

IID t70 net relchem change +5.511187e-02, with 53 identities worse and the largest 20 explaining 79.3% of the positive mass. DB-strat t70 net change +3.855787e-02, top-20 share 84.2%.

Candidate A asks whether an independent 90-update IID AdamW stream from the same corrected P536 reproduces a relchem ratio above 1 driven by ABDE4, pTC13, and PA8. It keeps the qualified singleton loss, the task coefficients, and the uniform identity sampler, so it does not change the intended training gradient. The earlier IID continuation averaged 15.3 s/update over 31 updates. The DB-strat run averaged 32.7 s/update over 90 updates. A new 90-update arm is therefore about 25-50 minutes on one local GPU. One fixed-variant chemistry audit plus one 90-system mRKS audit, using the existing evaluator, adds roughly 6 minutes if those receipts are collected. The main risk is a second negative that still does not identify a mechanism. The stream-specific hypothesis is falsified if the new checkpoint with the best Clean28 still has relchem/P536 at least 1, with positive deltas for ABDE4, pTC13, and PA8. It is supported only if all four scientific ratios are finite and strictly below 1. One bounded arm can test it.

Candidate B would emphasize databases with excess relchem loss, in the spirit of Skala's relative-excess update. Without an importance correction it changes the expected training gradient. The database-stratified arm already raised ABDE4 draws from 0 to 8, pTC13 from 4 to 15, and PA8 from 5 to 9, with the inverse-probability correction, and those three deltas stayed positive. Skala's baseline targets and normalizers for these eight databases are not available and are not invented here. B is a different experiment from A, and this postmortem does not justify running it next. Its cost would be another 90-update trajectory. It would be falsified if the fixed-panel relchem ratio did not fall relative to a paired IID replica.

Candidate C would change the qualified loss weights or the four task coefficients. That changes the objective whose eligibility is being judged. The present deltas are already inside the weighted singleton loss, and the reduced-learning-rate branch, which continued the same IID stream at 3e-5, still left ABDE4, pTC13, and PA8 above P536. A short coefficient trial could be bounded, but it would not answer whether the current objective can pass.

Selected next experiment: **A. One independent 90-update IID AdamW replica from corrected P536, with unchanged coefficients, the qualified singleton loss, and the existing uniform identity sampler.** Stop on a nonfinite update, a manifest or coefficient change, or a nonfinite scientific ratio. Do not promote a checkpoint unless relchem, AE17, Exc, and the operator are all strictly below their P536 values. Do not start B or C from this result. If A reproduces the same three-database pattern, the next question is why those qualified losses rise under AdamW, not another sampler.

## Input receipts

- `chemistry_dbstrat_t70`: `C:\Dev\readWFN_share_ms\lap_dbstrat_importance_20261009\endpoint_70_chemistry_one_variant.json` `dc6529d41f71012f83b8d1e374025cdac26192ad499017d131fcf6cc57754cc2`
- `chemistry_dbstrat_t90`: `C:\Dev\readWFN_share_ms\lap_dbstrat_importance_20261009\endpoint_90_chemistry_one_variant.json` `f3fc6bea222fc409cc3dccc04cffc2abc44cd915409eefbe33901eaafad9e5e0`
- `chemistry_iid_t59`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\endpoint_59_chemistry_one_variant.json` `ac757c8a7d7231e688c23bd566b336eb5385892b00797b42a6f36eb92035e3f1`
- `chemistry_iid_t70`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_lr_branches_20261009\A\endpoint_70_chemistry_one_variant.json` `1ae1b93705fd556c0b1d3cd53c478829c14b0b93a99dca5ee860737db93b27e8`
- `chemistry_iid_t80`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_lr_branches_20261009\A\endpoint_80_chemistry_one_variant.json` `a504ed0a7049ae90fc1806152d6a6e6a3b23fc52f38d7c137b9f8f955bfe3917`
- `chemistry_iid_t90`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\endpoint_90_chemistry_one_variant.json` `c71afe00df58d4e4371a671659a8bf5c40eebbe710818f5215331598979e54b0`
- `chemistry_lr3e-5_t80`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_lr_branches_20261009\B\endpoint_80_chemistry_one_variant.json` `771c84a334c952ae3cf6556815c93ebac0b690a82f329ecc25d94876a84975e0`
- `chemistry_t0`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\baseline_0_chemistry_one_variant.json` `3ada17228dc868beb1d7b453db49edb3a25d41fb6e5da1c2f698f3781fe355ba`
- `clean28_dbstrat_t70`: `C:\Dev\readWFN_share_ms\lap_dbstrat_importance_20261009\endpoint_70_validation.json` `934716033bd6d1634de0307d00464131e4ebca05f50cca9cb0dbe4a2cd5e8762`
- `clean28_iid_t70`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\endpoint_70_validation.json` `4c4fb14593ba3cccd5d490ef30bbde678ac06a94ff179b410964d10a923ca3b1`
- `clean28_lr3e-5_t80`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_lr_branches_20261009\B\endpoint_80_validation.json` `f000724b04f5bfc245b11087acabb47ee0e56de85bfe58f42ba2caeb46757a5d`
- `clean28_t0`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\baseline_0_validation.json` `dc8cc833ee30dc2771b8aed4e8445cf0b77b74bf64882d04be7bf8d8a33ca615`
- `dbstrat_metrics`: `C:\Dev\readWFN_share_ms\lap_full_vxc\lap_dbstrat_importance_metrics.json` `63e0687f6766957ccd81d420919b1f2c8d9516963fcf9927e344f5b3aab8e576`
- `dbstrat_sampling_manifest`: `C:\Dev\readWFN_share_ms\lap_dbstrat_importance_20261009\sampling_manifest.json` `5d540c6bef03b38bb6cc13c209cb23d2358afc4f90d42fcf4001edba8a285755`
- `diet30_reactions`: `C:\Dev\readWFN_share_ms\publication_dataset_v1\validation\diet30_reactions.jsonl` `426604af0fbd7e923337de31f74a74ac83534e11e7b68d2663f49bedcc8baa68`
- `evaluation_manifest`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\evaluation_manifest.json` `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`
- `iid_checkpoint_90`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\ordinary_sgd_adamw\checkpoint_90.pt` `4bc5a84aeea8653fdae2121d440e979376fffd7052f3b8713b0bdfd3e4103e7a`
- `iid_metrics`: `C:\Dev\readWFN_share_ms\lap_full_vxc\iid_adamw_t59_t90_metrics.json` `440b253810c9df320918765a9522fa930bc1e0b030c5b4167fb9451445e0bc76`
- `iid_sampling_manifest`: `C:\Dev\readWFN_share_ms\lap_iid_adamw_t59_t90_20261009\sampling_manifest.json` `e84e237d88449edfae5c68f7caecd85791b57f60a4b6239a37e1ae1350a71089`
- `lr_metrics`: `C:\Dev\readWFN_share_ms\lap_full_vxc\iid_adamw_lr_stabilization_metrics.json` `8e8723f844d29185b3837d7607ba76f73bd099e90f9d7e0696df975b9887feba`
