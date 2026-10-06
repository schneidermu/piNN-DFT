# Qualified q-Jacobian / scientific-objective coupling audit

**Mixed mechanism; initialization investigation is not closed.** The four objectives substantially engage the probe-resolved q-control space. Chemistry wants q-response changes opposite to AE17/Exc/operator in both states. However, conflict also occurs in the orthogonal complement. Predopt depth changes chemistry's coupling much more than it changes the dominant q-control direction. No single A–E case meets every frozen descriptive condition; CASE E closure is specifically unsupported.

## Frozen scope and provenance

Starting commit `61278273c7c167c51372ebd9dfd17028e2e38fcd`, branch `lap_full_vxc`; final commit is the Git commit containing this report. Production-source changes: zero. Only seed11 P67/P536 and their stored arrays were analyzed.
The existing qualified exact128x9446 Jacobians/SVDs were reused by SHA. No Jacobian, model forward, objective loss or scientific gradient was recomputed. The same64point probe, h2, alpha64/beta64 ordering and F64 q-observable on widened F32 state/source values remain unchanged.
Stored gradients are +grad L. Chemistry uses the established matched-F64 shadow; Exc/operator use their established production paths with stored F32 leaves and gradient values widened after autograd. This audit does not upgrade their numerical precision. Induced response is **Jq(-g/||g||)**, a unit parameter-direction first-order gain; no finite model update is applied.
| State | model-state SHA256 | gradient file SHA256 | exact J/SVD file SHA256 |
| --- | --- | --- | --- |
| P67 | ee815caf918e8ded13e99896070a95a32293100499746fd0623444e2c0023523 | d79c78b34e81a44f6a5ea1bebba1283bbf293c522e662f0c750b3cbcf425a381 | 8b1d825cd8871c2a247ae2494815e68d65936640e4631609d91a7f731ab865dc |
| P536 | 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6 | e7c4c07f53dc4b1598c29d29bb1e51a1704c22474c29f1ecd3070bdfc611a7e5 | 7a50fcaac9f42063c822aeffa2914c850bec2816c065e93da8c44ec6cddedc91 |

Ordering compatibility PASS: all four gradient tensors are finite F64 vectors of length9446; their norms reproduce SHA-bound receipts within1e-12 relative error. Legacy tensors have no embedded parameter-name manifest. Ordering is proven from exact executed matrix.py SHA: flat() zips autograd gradients with params.values(), where named_trainable_parameters selects unique requires_grad parameters and sorts names. Both main and chemistry-shadow use this helper. Its SHA and the exact-J canonical34-tensor manifest are verified; no realignment or guess is used. All state IDs and original file SHA bindings match.
Objectives: full251 mean corrected fchem over non-AE reactions; full17 AE mean; equal-system mean Exc and operator over the same15 systems. No R2 gradients are used. Parent receipts retain the canonical source/catalog identities.
Protocol was frozen before coupling results. A pre-execution amendment added only a justified Ruff annotation for extracting the existing SHA-verified pure MGDA solver. One initial synthetic-test process aborted under the default Windows MKL/PyTorch runtime; unchanged tests passed under the established MKL_THREADING_LAYER=SEQUENTIAL, OMP_NUM_THREADS=1 runtime, recorded before scientific analysis. No result or sample was selected from that failure.

## Projection definition and safeguards

B = first13 rows of exact Vh satisfying sigma/sigma1>=1e-3, separately per state. gq=g B^T B and gperp=g-gq. Squared projected fractions use ||gq||²/||g||². Absolute norms accompany fractions. Component cosines are undefined when component norm<=1e-8 of its original full gradient norm; no arbitrary absolute denominator epsilon is used.
This is a **probe-resolved q-control subspace**, not a parameter space that changes only q physics. A q-control direction may also alter rho/sigma dependence. The complement is not q-null: lower-gain discarded singular directions can still affect q. Jgperp response norms are explicitly retained. Equal128-row Euclidean aggregation gives64 alpha and64 beta rows without changing their separately frozen q convention.

## P67 objective projections

| Objective | ||g|| | fraction top1 | fraction top2 | fraction top3 | fraction top4 | fraction top8 | fraction topfull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | 10.9412956522 | 0.0145627305765 | 0.0249806527346 | 0.0687478571765 | 0.0691862254907 | 0.166572047256 | 0.201972239044 |
| ae17 | 5540.57721116 | 0.0789857575666 | 0.207305651759 | 0.260686930104 | 0.332350656728 | 0.374976582551 | 0.396134543879 |
| exc | 15019.1896692 | 0.0950526586867 | 0.216056852974 | 0.269750374725 | 0.341781317944 | 0.383934249146 | 0.404712924644 |
| op | 1.07912633937 | 0.895437888345 | 0.89762105513 | 0.897657149194 | 0.900759030756 | 0.906858425878 | 0.910368878007 |

| Objective | projected norm top1 | projected norm top2 | projected norm top3 | projected norm top4 | projected norm top8 | projected norm topfull |
| --- | --- | --- | --- | --- | --- | --- |
| full251 | 1.32035331735 | 1.72930120709 | 2.86878721389 | 2.87791904065 | 4.46549713737 | 4.91716287024 |
| ae17 | 1557.14625409 | 2522.67081361 | 2828.88074934 | 3194.13510598 | 3392.79082299 | 3487.19609102 |
| exc | 4630.50797207 | 6981.20714969 | 7800.59140456 | 8780.52860079 | 9306.25459443 | 9554.76563252 |
| op | 1.02115113933 | 1.02239521561 | 1.02241577107 | 1.02418074204 | 1.0276424579 | 1.02962954053 |

Signed coefficients a_k=v_k^T g in this state's stored exact-SVD sign convention:
| Objective | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | -1.32035331735 | -1.11675860516 | -2.28898611044 | -0.22908061017 | -1.42786856349 | -1.28713210359 | 1.08340300376 | -2.60556460753 |
| ae17 | 1557.14625409 | 1984.73262109 | 1280.11650258 | 1483.21703782 | -153.421893157 | 526.551635338 | -287.283288294 | 961.875201659 |
| exc | 4630.50797207 | 5224.523824 | 3480.22599754 | 4030.93739078 | -341.621804421 | 1423.24279195 | -770.82107222 | 2602.34537285 |
| op | 1.02115113933 | -0.0504214987438 | 0.00648321144471 | 0.0601014428991 | -0.00250321664888 | -0.00349796046556 | 0.0117499627199 | 0.0833442587725 |

Signed unit-gradient coefficients a_k/||g||:
| Objective | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | -0.120676139218 | -0.102068223057 | -0.209206129073 | -0.0209372470549 | -0.13050269446 | -0.117639824799 | 0.0990196260299 | -0.238140407715 |
| ae17 | 0.281044049157 | 0.35821766315 | 0.231043888351 | 0.267700815509 | -0.0276905974431 | 0.0950355198151 | -0.0518507869026 | 0.173605594688 |
| exc | 0.308306111984 | 0.34785657143 | 0.23171862625 | 0.268385810391 | -0.0227456881459 | 0.0947616231828 | -0.0513224141379 | 0.17326802778 |
| op | 0.946275799302 | -0.0467243703582 | 0.00600783356702 | 0.0556945379855 | -0.00231966967867 | -0.00324147445758 | 0.0108884032307 | 0.0772330872967 |

Dominant modes; coefficients are gradients, so the negative-gradient step reverses their signs:
| Mode | sigma | fraction of full J energy | full251 | ae17 | exc | op |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 53.9733594644 | 0.974402320349 | -1.32035331735 | 1557.14625409 | 4630.50797207 | 1.02115113933 |
| 2 | 6.75603266166 | 0.0152673228338 | -1.11675860516 | 1984.73262109 | 5224.523824 | -0.0504214987438 |
| 3 | 4.41069255811 | 0.00650718242086 | -2.28898611044 | 1280.11650258 | 3480.22599754 | 0.00648321144471 |

Induced q-slope response from -g/||g||:
| Objective | L2 | RMS | median abs | alpha L2 | beta L2 | max abs | J-perp L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | 6.61924261689 | 0.58506391759 | 0.00292640491256 | 4.88995824696 | 4.46124211002 | 4.46982901404 | 0.0043407168174 |
| ae17 | 15.4131640482 | 1.36234410225 | 0.00114537837052 | 11.3903905348 | 10.3838638975 | 10.3834282379 | 0.00308311328581 |
| exc | 16.853632244 | 1.48966470592 | 0.0011231549566 | 12.4549948955 | 11.3542072365 | 11.3620671971 | 0.00305988711854 |
| op | 51.0749109981 | 4.51442698941 | 0.000602695524962 | 38.0628793839 | 34.0567724025 | 34.1808800274 | 0.00115426247942 |

| Pair | full gradient cosine | resolved-q cosine | complement cosine | induced q-response cosine |
| --- | --- | --- | --- | --- |
| full251__ae17 | -0.879580428546 | -0.716140348574 | -0.975257264197 | -0.994535285397 |
| full251__exc | -0.878210112849 | -0.718117150143 | -0.97628581562 | -0.995045495477 |
| full251__op | -0.401501691207 | -0.323936269398 | -0.981867359622 | -0.983429435828 |
| ae17__exc | 0.999534587769 | 0.998959269563 | 0.999982673634 | 0.999816944126 |
| ae17__op | 0.518858298821 | 0.47716755354 | 0.998537857753 | 0.983341036062 |
| exc__op | 0.543448658018 | 0.515292286982 | 0.998619731262 | 0.986625657445 |

Additive conflict accounting, normalized by original full-gradient norms: full cosine = q dot contribution + complement dot contribution. The negative-part fraction counts q's share of opposed contributions without hiding positive cancellation.
| Pair | q dot contribution | complement dot contribution | q share of negative contributions |
| --- | --- | --- | --- |
| full251__ae17 | -0.202565435067 | -0.677014993479 | 0.2302977971 |
| full251__exc | -0.205312161736 | -0.672897951113 | 0.233784784224 |
| full251__op | -0.138903797017 | -0.262597894189 | 0.345960677276 |
| ae17__exc | 0.399984052235 | 0.599550535534 | undefined/tiny |
| ae17__op | 0.286550363584 | 0.232307935237 | undefined/tiny |
| exc__op | 0.312777770823 | 0.230670887196 | undefined/tiny |

Optional component-unit MGDA / max-min diagnostic:
| Subspace | gamma | MGDA coefficients | max KKT certificate residual |
| --- | --- | --- | --- |
| q | 0.369606316115 | [0.47828093489719353, 0.4402513577229284, 0.0, 0.0814677073798782] | 2.22044604925e-16 |
| complement | 0.0952172263258 | [0.49999999999999983, 0.0, 0.0, 0.5000000000000002] | 1.66533453694e-16 |

Positive gamma certifies a shared first-order descent direction inside this projected component space (apply negative of the positive gradient-space direction). This is not a finite-step acceptance test, optimizer recommendation or global convergence claim.

## P536 objective projections

| Objective | ||g|| | fraction top1 | fraction top2 | fraction top3 | fraction top4 | fraction top8 | fraction topfull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | 9.62453223792 | 0.407932124535 | 0.407932383422 | 0.432612957931 | 0.43318734766 | 0.502095835923 | 0.512430865147 |
| ae17 | 5545.38544457 | 0.0826593191729 | 0.214280246645 | 0.340156047273 | 0.352610445809 | 0.398529960467 | 0.408624615856 |
| exc | 15023.3743578 | 0.0979245342517 | 0.223169563378 | 0.34815866577 | 0.360936576452 | 0.406407488683 | 0.41627281878 |
| op | 0.629643090783 | 0.709842027276 | 0.713253364685 | 0.734534807777 | 0.756468286887 | 0.780873100688 | 0.783200420118 |

| Objective | projected norm top1 | projected norm top2 | projected norm top3 | projected norm top4 | projected norm top8 | projected norm topfull |
| --- | --- | --- | --- | --- | --- | --- |
| full251 | 6.14714680739 | 6.14714875798 | 6.33037435479 | 6.33457544933 | 6.81982045789 | 6.88965177537 |
| ae17 | 1594.3279146 | 2566.98190305 | 3234.22951633 | 3292.9059365 | 3500.75909844 | 3544.81847755 |
| exc | 4701.24891936 | 7097.16612796 | 8864.53775306 | 9025.74244723 | 9577.41574855 | 9692.96213523 |
| op | 0.530487672905 | 0.531760845924 | 0.539635649627 | 0.547633245331 | 0.556396863863 | 0.557225391459 |

Signed coefficients a_k=v_k^T g in this state's stored exact-SVD sign convention:
| Objective | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | -6.14714680739 | -0.0048970514815 | -1.51201905379 | 0.230665670525 | -1.33570814793 | -0.491092233082 | -0.118667811391 | 2.08416291765 |
| ae17 | 1594.3279146 | 2011.84358023 | 1967.44618064 | 618.861004049 | -68.2366316129 | 437.428715056 | 28.5996528989 | -1102.39135513 |
| exc | 4701.24891936 | 5316.76834609 | 5311.33340391 | 1698.23353764 | -121.067239788 | 1172.85830918 | 84.1490555328 | -2977.50411306 |
| op | 0.530487672905 | -0.0367753470998 | 0.0918533455558 | 0.0932498635071 | 0.0094350852559 | 0.0142736470826 | 0.0221235019177 | -0.0943031895147 |

Signed unit-gradient coefficients a_k/||g||:
| Objective | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | -0.638695643116 | -0.000508809296955 | -0.15710052358 | 0.0239664292064 | -0.138781617112 | -0.0510250494198 | -0.0123297224693 | 0.216546930919 |
| ae17 | 0.287505337642 | 0.36279598602 | 0.354789797806 | 0.111599276594 | -0.0123051196882 | 0.0788815708895 | 0.00515737872232 | -0.198794360852 |
| exc | 0.312928960391 | 0.353899744456 | 0.353537978713 | 0.113039420918 | -0.00805859169216 | 0.0780688999188 | 0.00560120872506 | -0.198191434371 |
| op | 0.842521232537 | -0.0584066555136 | 0.145881606422 | 0.148099558102 | 0.0149848150389 | 0.0226694254119 | 0.0351365753735 | -0.149772451878 |

Dominant modes; coefficients are gradients, so the negative-gradient step reverses their signs:
| Mode | sigma | fraction of full J energy | full251 | ae17 | exc | op |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 47.9478522062 | 0.973749590245 | -6.14714680739 | 1594.3279146 | 4701.24891936 | 0.530487672905 |
| 2 | 6.20022199681 | 0.0162825886375 | -0.0048970514815 | 2011.84358023 | 5316.76834609 | -0.0367753470998 |
| 3 | 4.03859524358 | 0.00690827521478 | -1.51201905379 | 1967.44618064 | 5311.33340391 | 0.0918533455558 |

Induced q-slope response from -g/||g||:
| Objective | L2 | RMS | median abs | alpha L2 | beta L2 | max abs | J-perp L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | 30.6312261357 | 2.70744346458 | 0.00184043272547 | 22.7809166392 | 20.4768613722 | 20.5194463174 | 0.00206361498635 |
| ae17 | 14.0432693851 | 1.24126137653 | 0.00243311817643 | 10.3949840702 | 9.44233663881 | 9.40309145486 | 0.00221464577402 |
| exc | 15.23318462 | 1.34643601798 | 0.00239319216627 | 11.2743537951 | 10.2439670133 | 10.2143865954 | 0.00220425867289 |
| op | 40.4043671279 | 3.57127524821 | 0.00117318887573 | 30.1014446791 | 26.9521040225 | 26.9945595077 | 0.00137636270217 |

| Pair | full gradient cosine | resolved-q cosine | complement cosine | induced q-response cosine |
| --- | --- | --- | --- | --- |
| full251__ae17 | -0.815951040759 | -0.632414129883 | -0.980618601043 | -0.983502133702 |
| full251__exc | -0.829247264938 | -0.662065181337 | -0.98122236228 | -0.986676201976 |
| full251__op | -0.918322701424 | -0.943059881464 | -0.986962646162 | -0.999877891344 |
| ae17__exc | 0.999605224713 | 0.999133530548 | 0.999987671799 | 0.999828939699 |
| ae17__op | 0.68261873462 | 0.575460523688 | 0.997227222037 | 0.981650685782 |
| exc__op | 0.702239741395 | 0.608494392442 | 0.997348451984 | 0.985008881366 |

Additive conflict accounting, normalized by original full-gradient norms: full cosine = q dot contribution + complement dot contribution. The negative-part fraction counts q's share of opposed contributions without hiding positive cancellation.
| Pair | q dot contribution | complement dot contribution | q share of negative contributions |
| --- | --- | --- | --- |
| full251__ae17 | -0.289388632583 | -0.526562408176 | 0.35466421161 |
| full251__exc | -0.305778830813 | -0.523468434125 | 0.368742646182 |
| full251__op | -0.597438720547 | -0.320883980877 | 0.650576011701 |
| ae17__exc | 0.41207363012 | 0.587531594593 | undefined/tiny |
| ae17__op | 0.325547417952 | 0.357071316668 | undefined/tiny |
| exc__op | 0.347441797469 | 0.354797943925 | undefined/tiny |

Optional component-unit MGDA / max-min diagnostic:
| Subspace | gamma | MGDA coefficients | max KKT certificate residual |
| --- | --- | --- | --- |
| q | 0.154564179441 | [0.4856562026231029, 0.0, 0.08288726982688555, 0.4314565275500113] | 2.22044604925e-16 |
| complement | 0.0807383237323 | [0.5, 0.0, 0.0, 0.5] | 2.22044604925e-16 |

Positive gamma certifies a shared first-order descent direction inside this projected component space (apply negative of the positive gradient-space direction). This is not a finite-step acceptance test, optimizer recommendation or global convergence claim.

## Matched-depth subspace and coupling comparison

Principal-angle comparison is valid in shared seed11 lineage coordinates; no different seeds or hidden-unit permutations are mixed. Arbitrary SVD signs do not affect subspace/projection metrics. Cross-state corresponding-mode coefficient signs are aligned by each P67/P536 mode dot product; raw within-state coefficients remain intact. Corresponding low-mode vectors can rotate, so individual-index comparisons remain descriptive.
| Top k | canonical correlations | largest angle degrees | mean squared overlap |
| --- | --- | --- | --- |
| 1 | 0.993258746271 | 6.6565966668 | 0.986562937044 |
| 2 | 0.993685884553, 0.882002350638 | 28.1151428041 | 0.882669891845 |
| 3 | 0.994565145145, 0.910111580535, 0.904652640306 | 25.2234598896 | 0.878619772192 |
| 4 | 0.998254899379, 0.99021174138, 0.930813026706, 0.890053012199 | 27.1200908738 | 0.908909848028 |
| 8 | 0.998506414774, 0.995988698558, 0.989722408031, 0.989202680789, 0.963943152196, 0.925870721617, 0.902619987785, 0.840742004093 | 32.7814434417 | 0.906884236281 |
| full | 0.998832908382, 0.997961995307, 0.994086775848, 0.992979557836, 0.979280458353, 0.973545234845, 0.966719090542, 0.960308295462, 0.95483646765, 0.94233234681, 0.916970775132, 0.897368545538, 0.811251773053 | 35.7815860615 | 0.910405283534 |

| Mode | abs corresponding vector cosine | P536 sign alignment | P67 unit coefficients (chem/AE/Exc/op) | P536 aligned unit coefficients |
| --- | --- | --- | --- | --- |
| 1 | 0.993258746271 | 1 | -0.120676139218, 0.281044049157, 0.308306111984, 0.946275799302 | -0.638695643116, 0.287505337642, 0.312928960391, 0.842521232537 |
| 2 | 0.881721491554 | 1 | -0.102068223057, 0.35821766315, 0.34785657143, -0.0467243703582 | -0.000508809296955, 0.36279598602, 0.353899744456, -0.0584066555136 |
| 3 | 0.878128640043 | 1 | -0.209206129073, 0.231043888351, 0.23171862625, 0.00600783356702 | -0.15710052358, 0.354789797806, 0.353537978713, 0.145881606422 |
| 4 | 0.89671940926 | 1 | -0.0209372470549, 0.267700815509, 0.268385810391, 0.0556945379855 | 0.0239664292064, 0.111599276594, 0.113039420918, 0.148099558102 |
| 5 | 0.898474018325 | 1 | -0.13050269446, -0.0276905974431, -0.0227456881459, -0.00231966967867 | -0.138781617112, -0.0123051196882, -0.00805859169216, 0.0149848150389 |
| 6 | 0.876944265244 | 1 | -0.117639824799, 0.0950355198151, 0.0947616231828, -0.00324147445758 | -0.0510250494198, 0.0788815708895, 0.0780688999188, 0.0226694254119 |
| 7 | 0.921991752719 | 1 | 0.0990196260299, -0.0518507869026, -0.0513224141379, 0.0108884032307 | -0.0123297224693, 0.00515737872232, 0.00560120872506, 0.0351365753735 |
| 8 | 0.81636800729 | -1 | -0.238140407715, 0.173605594688, 0.17326802778, 0.0772330872967 | -0.216546930919, 0.198794360852, 0.198191434371, 0.149772451878 |

| Objective | P67 norm | P536 norm | P67 q fraction | P536 q fraction | P67 projected norm | P536 projected norm | P536/P67 induced gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full251 | 10.9412956522 | 9.62453223792 | 0.201972239044 | 0.512430865147 | 4.91716287024 | 6.88965177537 | 4.62760287069 |
| ae17 | 5540.57721116 | 5545.38544457 | 0.396134543879 | 0.408624615856 | 3487.19609102 | 3544.81847755 | 0.911121774942 |
| exc | 15019.1896692 | 15023.3743578 | 0.404712924644 | 0.41627281878 | 9554.76563252 | 9692.96213523 | 0.903851727594 |
| op | 1.07912633937 | 0.629643090783 | 0.910368878007 | 0.783200420118 | 1.02962954053 | 0.557225391459 | 0.791080519541 |

Chemistry's normalized leading-mode coefficient changes from -0.120676 to -0.638696 while the leading singular vectors have abs cosine0.993259 and sigma1 declines only about11%. Thus its4.63x normalized induced response change is not mere uniform singular-value rescaling. AE/Exc dominant-mode coupling and induced-response preferences are much more similar across depth. This does not imply fewer accessible q directions; it describes changed objective engagement at different functional points.
Mode1 carries about97.4% of J energy in each state and has opposite signed chemistry vs AE/Exc/operator coefficients. This explains strong opposition after mapping into q-response space. Full-gradient opposition is not entirely in this mode/subspace: at P67, q accounts for23–35% of chemistry-secondary negative contributions; at P53635–65%, with operator65%. Complement chemistry-secondary cosines remain roughly-0.98. Weak q modes and complement directions permit positive common-descent margins despite dominant-response opposition.

## Frozen threshold sensitivity and classification

| P67 relative singular threshold | rank | full251 | ae17 | exc | op |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 6 | 0.100056307131 | 0.342149175941 | 0.351278449502 | 0.90077491878 |
| 0.0001 | 28 | 0.258451189644 | 0.438237044412 | 0.446318616729 | 0.916008922665 |

| P536 relative singular threshold | rank | full251 | ae17 | exc | op |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 6 | 0.455051240577 | 0.358984164005 | 0.367096270486 | 0.757206734417 |
| 0.0001 | 24 | 0.528515300345 | 0.425133687396 | 0.432627186429 | 0.789270846182 |

Primary threshold remains1e-3. The fixed1e-2/1e-4 diagnostics show projection fractions depend on how many weaker modes are retained; chemistry's stronger P536 coupling persists at both alternatives. No cutoff was tuned. Induced q-response vectors use the entire exact J, so their opposition/gain comparisons do not depend on the retained-subspace threshold.
Frozen descriptive classifier result: **mixed**. CASE E fails because chemistry projection changes by0.310459 and induced gain4.63x, exceeding the predeclared similarity cutoffs. Strict CASE D's strong-rotation criterion is not met: leading abs cosine0.993259 and full mean-squared overlap0.910405 remain high, despite some principal angles reaching35.78degrees. CASE A majority-q conflict is supported for P536 chemistry/operator, but not uniformly across both states. CASE B/C are contradicted by substantial chemistry/secondary q projections. We do not relax cutoffs or force a single A–E label post hoc.
Scientific synthesis: contested q modes are part of the MOO conflict, and predopt depth materially changes **objective coupling** while retaining a substantially shared q-control subspace. This is mixed evidence: P536-only A-like chemistry/operator conflict plus depth-sensitive coupling that does not meet D's rotation gate. It is not proof of a predopt-induced capacity bottleneck. Initialization investigation is not closed, and optimizer choice is not made here. Seed23 expansion is not triggered under the frozen CASE D rule; no gate is relaxed despite the large coupling change.
ONE next experiment, not launched: a read-only same-array analysis of the already-certified resolved-q common-descent directions and their induced q-mode response composition. This would clarify how weaker modes permit common descent despite strongly opposed dominant-mode responses, without new seeds, initialization states, gradients or training. It must not select an optimizer from this diagnostic alone.

## Validation, review and safety

Five focused array-analysis tests PASS, including exact projection fractions/norms, negative-step response sign, relative tiny-component handling, sign-invariant subspace overlap, additive cosine decomposition, and the reused common-descent certificate/update sign. Ruff, py_compile and git diff --check PASS. All32 immutable hashes match before/after; state mutationNONE; ordering compatibilityPASS. No model was loaded for forward evaluation; no checkpoint, EMA, RNG/cursor or optimizer state was changed. Existing large arrays remain external and are SHA-bound; no dense data is copied into Git.
Independent review: PASS. Review evidence and limitations are recorded in metrics; it must challenge provenance, signs, q/perp attribution, shared-lineage overlap, scalar/common-descent interpretation and case boundaries.
No training trajectory, production optimizer change, old cursor10 advancement, new initialization variant, unfinished initialization-matrix resumption, expensive scientific-gradient recomputation, full90, SCF, Diet, Slurm or100-update run occurred. No follow-up was launched.
