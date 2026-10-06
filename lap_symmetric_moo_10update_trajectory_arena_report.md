# Symmetric MOO ten-update trajectory Arena

Starting commit: `e0faea894a29e169ef29fc319e39b43a5ee6cc57`. Final commit: the Git commit containing this report.

**CLEAR UNIT_MAXMIN WIN** under the frozen relative-method promotion rule. Both methods have zero scientific passes: chemistry rose at every t10 endpoint. UNIT_MAXMIN is the relative production candidate for the next bounded qualification, not a scientifically qualified production trainer.

Protocol SHA: `65c0336fd17a9f5b3dab5c76d794cc0b7726253b4a6237897114476a7e56a743`. Eight arms were attempted with maximum ten update opportunities each. No competitive pruning at t5. A rejected/failed attempt stops its arm without advancing committed state.

Manifest byte SHA: `eb64fb2ba9a98eead1dc518a1b0853eeb90d66dcb92c117dbfba23964e6afcfc`. Actual canonical SHA: `2e87f7e79a66b425628bdf59a821f785e05059f193a492d4a4f3130d7411e9f3`. The supplied label contained `628fdf` instead of the byte-identical historical file’s `628bdf`; exact requested file-byte SHA matched. This discrepancy was disclosed before execution. No sampling stream was regenerated.

Starting tensor-state identities:
| Start | State SHA256 |
| --- | --- |
| 11_P67 | ee815caf918e8ded13e99896070a95a32293100499746fd0623444e2c0023523 |
| 11_P536 | 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6 |
| 23_P67 | 69f6e7c90a5f4cf6f51e885b8f3c1b9679508c71a57a8363149de6783820e4d8 |
| 23_P536 | 0b0c2aefc703c90be250c2fdb79e60be1df35273c7bc7d680ede50a8e7145074 |

Four optimized scalar objectives: full251 corrected relative-chemistry mean; full17 corrected AE loss mean; equal-system mean Exc; equal-system mean operator. RMSE/MAE are secondary human-readable metrics, never substituted for the optimized AE scalar.

Each method uses a global unit parameter direction and the same start-specific eta0 from historical cold PCD_CONTROL. Existing repaired operator and chemistry precision paths are reused. The controller API tag `pcd` is solely compatibility plumbing for its custom-aggregator hook; the candidate direction is UNIT_MAXMIN or unchanged Nash-MTL. No production source behavior changed.

Frozen eta0 values:
| Start | eta0 |
| --- | --- |
| 11_P67 | 7.2568949448274828e-06 |
| 11_P536 | 6.3835419098504467e-06 |
| 23_P67 | 8.5392826698708043e-06 |
| 23_P536 | 1.0204187208883716e-05 |

Arm status:
| Arm | Status | Accepted | Scientific pass | Accepted t | Backtracks |
| --- | --- | --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 11_P67_Nash-MTL | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 11_P536_UNIT_MAXMIN | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 11_P536_Nash-MTL | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 23_P67_UNIT_MAXMIN | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 23_P67_Nash-MTL | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 23_P536_UNIT_MAXMIN | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 23_P536_Nash-MTL | TRAJECTORY-PASS | 10 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |

Absolute optimized endpoints:
| Arm | Cursor | Chemistry | AE17 loss | Exc | Operator |
| --- | --- | --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | 0 | 1.237027054764855 | 29.08069447031722 | 80.9058999863264 | 0.04142061014895267 |
| 11_P67_UNIT_MAXMIN | 5 | 1.237028371476554 | 29.01372087702236 | 80.7219684489627 | 0.04140617473974769 |
| 11_P67_UNIT_MAXMIN | 10 | 1.237060008508506 | 28.9359972929692 | 80.5026052493933 | 0.04138098124219735 |
| 11_P67_Nash-MTL | 0 | 1.237027054764855 | 29.08069447031722 | 80.9058999863264 | 0.04142061014895267 |
| 11_P67_Nash-MTL | 5 | 1.237094928825659 | 28.9885824402407 | 80.65348209047472 | 0.04140361563578369 |
| 11_P67_Nash-MTL | 10 | 1.237174621572105 | 28.89532910336415 | 80.3896993304003 | 0.04137263560376388 |
| 11_P536_UNIT_MAXMIN | 0 | 1.237333214793654 | 24.87457454408775 | 69.98481862305971 | 0.03686135457451291 |
| 11_P536_UNIT_MAXMIN | 5 | 1.237361061497784 | 24.82153401651828 | 69.84410839695228 | 0.03685845824356131 |
| 11_P536_UNIT_MAXMIN | 10 | 1.237491323503801 | 24.74730585763693 | 69.63652889819009 | 0.03684614070158898 |
| 11_P536_Nash-MTL | 0 | 1.237333214793654 | 24.87457454408775 | 69.98481862305971 | 0.03686135457451291 |
| 11_P536_Nash-MTL | 5 | 1.237382980035784 | 24.79516974817301 | 69.77467163978118 | 0.03685865289842751 |
| 11_P536_Nash-MTL | 10 | 1.237558535625611 | 24.71058174923715 | 69.53792109633427 | 0.03684348937907618 |
| 23_P67_UNIT_MAXMIN | 0 | 1.244320646100938 | 23.18721186424887 | 65.15294469673954 | 0.03749161497138206 |
| 23_P67_UNIT_MAXMIN | 5 | 1.24435088815872 | 23.03482730557512 | 64.74982731636851 | 0.03748218153854389 |
| 23_P67_UNIT_MAXMIN | 10 | 1.244456314092326 | 22.87304559880458 | 64.3102126325189 | 0.0374669204522255 |
| 23_P67_Nash-MTL | 0 | 1.244320646100938 | 23.18721186424887 | 65.15294469673954 | 0.03749161497138206 |
| 23_P67_Nash-MTL | 5 | 1.244388169515735 | 23.00613879360202 | 64.67035676772502 | 0.03747982080532076 |
| 23_P67_Nash-MTL | 10 | 1.244539681333144 | 22.81854826165698 | 64.16287421443258 | 0.03746233358679234 |
| 23_P536_UNIT_MAXMIN | 0 | 1.250805907035362 | 24.89464795349126 | 70.05677837468198 | 0.03616181439536121 |
| 23_P536_UNIT_MAXMIN | 5 | 1.250747445574947 | 24.78092458177476 | 69.75388743963552 | 0.03615437467777412 |
| 23_P536_UNIT_MAXMIN | 10 | 1.250809988152748 | 24.60053436928673 | 69.26608093200896 | 0.0361379952457339 |
| 23_P536_Nash-MTL | 0 | 1.250805907035362 | 24.89464795349126 | 70.05677837468198 | 0.03616181439536121 |
| 23_P536_Nash-MTL | 5 | 1.250831917955088 | 24.73064268314334 | 69.61970953822589 | 0.03615164861780116 |
| 23_P536_Nash-MTL | 10 | 1.250953701393832 | 24.52319917046734 | 69.05912370327312 | 0.03613281691940746 |

Own-baseline ratios:
| Arm | Cursor | Chemistry | AE17 | Exc | Operator | Rmax |
| --- | --- | --- | --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | 0 | 1 | 1 | 1 | 1 | 1 |
| 11_P67_UNIT_MAXMIN | 5 | 1.00000106441625 | 0.9976969740745627 | 0.9977265992048195 | 0.9996514921158073 | 1.00000106441625 |
| 11_P67_UNIT_MAXMIN | 10 | 1.000026639468817 | 0.9950242874187302 | 0.9950152617176095 | 0.9990432563254666 | 1.000026639468817 |
| 11_P67_Nash-MTL | 0 | 1 | 1 | 1 | 1 | 1 |
| 11_P67_Nash-MTL | 5 | 1.000054868695509 | 0.9968325367824162 | 0.9968801052099496 | 0.9995897087679814 | 1.000054868695509 |
| 11_P67_Nash-MTL | 10 | 1.00011929149543 | 0.9936258273631576 | 0.9936197402659962 | 0.998841771161355 | 1.00011929149543 |
| 11_P536_UNIT_MAXMIN | 0 | 1 | 1 | 1 | 1 | 1 |
| 11_P536_UNIT_MAXMIN | 5 | 1.000022505420365 | 0.9978676810139823 | 0.9979894178641042 | 0.9999214263560026 | 1.000022505420365 |
| 11_P536_UNIT_MAXMIN | 10 | 1.000127781836176 | 0.9948835833865121 | 0.9950233531825593 | 0.9995872676655663 | 1.000127781836176 |
| 11_P536_Nash-MTL | 0 | 1 | 1 | 1 | 1 | 1 |
| 11_P536_Nash-MTL | 5 | 1.00004021975773 | 0.9968077928016819 | 0.9969972490118122 | 0.9999267070861453 | 1.00004021975773 |
| 11_P536_Nash-MTL | 10 | 1.000182101982929 | 0.9934072120686953 | 0.9936143647219772 | 0.9995153407778161 | 1.000182101982929 |
| 23_P67_UNIT_MAXMIN | 0 | 1 | 1 | 1 | 1 | 1 |
| 23_P67_UNIT_MAXMIN | 5 | 1.000024304071364 | 0.9934280775297225 | 0.9938127527121394 | 0.9997483855297945 | 1.000024304071364 |
| 23_P67_UNIT_MAXMIN | 10 | 1.000109029768022 | 0.9864508821809368 | 0.9870653265459725 | 0.9993413322105379 | 1.000109029768022 |
| 23_P67_Nash-MTL | 0 | 1 | 1 | 1 | 1 | 1 |
| 23_P67_Nash-MTL | 5 | 1.000054265285246 | 0.9921908217466184 | 0.9925929989617389 | 0.9996854185643828 | 1.000054265285246 |
| 23_P67_Nash-MTL | 10 | 1.000176027965857 | 0.9841005635024059 | 0.9848039027719263 | 0.9992189884428271 | 1.000176027965857 |
| 23_P536_UNIT_MAXMIN | 0 | 1 | 1 | 1 | 1 | 1 |
| 23_P536_UNIT_MAXMIN | 5 | 0.9999532609655207 | 0.9954318144233668 | 0.9956764935232031 | 0.999794265920793 | 0.9999532609655207 |
| 23_P536_UNIT_MAXMIN | 10 | 1.000003262790304 | 0.988185670078404 | 0.9887134769680078 | 0.9993413176295057 | 1.000003262790304 |
| 23_P536_Nash-MTL | 0 | 1 | 1 | 1 | 1 | 1 |
| 23_P536_Nash-MTL | 5 | 1.0000207953285 | 0.993412026928265 | 0.9937612198762762 | 0.999718880876692 | 1.0000207953285 |
| 23_P536_Nash-MTL | 10 | 1.000118159306443 | 0.9850791710845692 | 0.9857593412863899 | 0.9991981188876002 | 1.000118159306443 |

Human-readable metrics (secondary):
| Arm | Cursor | Jrel | NonAE RMSE | AE MAE | AE RMSE | Exc median | Operator median | System wins Exc/op |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | 0 | 1.237027054765 | 46.01917263444 | 55.05485292324 | 65.80519195214 | 65.63818613808 | 0.04020147300376 | None |
| 11_P67_UNIT_MAXMIN | 5 | 1.237028371477 | 46.01814594337 | 54.92806016964 | 65.64945895008 | 65.47932035855 | 0.04019181444286 | {'exc': 15, 'op': 15} |
| 11_P67_UNIT_MAXMIN | 10 | 1.237060008509 | 46.01830014224 | 54.78091579889 | 65.47182256717 | 65.27631621598 | 0.04017420619177 | {'exc': 15, 'op': 15} |
| 11_P67_Nash-MTL | 0 | 1.237027054765 | 46.01917263444 | 55.05485292324 | 65.80519195214 | 65.63818613808 | 0.04020147300376 | None |
| 11_P67_Nash-MTL | 5 | 1.237094928826 | 46.01998260899 | 54.88046870166 | 65.5903116303 | 65.423138072 | 0.04018988424611 | {'exc': 15, 'op': 15} |
| 11_P67_Nash-MTL | 10 | 1.237174621572 | 46.02166798772 | 54.70392378621 | 65.37731977833 | 65.17805705657 | 0.04016823350606 | {'exc': 15, 'op': 15} |
| 11_P536_UNIT_MAXMIN | 0 | 1.237333214794 | 46.1191207968 | 47.09193050568 | 56.46606900362 | 53.77353949162 | 0.03757695455659 | None |
| 11_P536_UNIT_MAXMIN | 5 | 1.237361061498 | 46.11989835071 | 46.99151548817 | 56.34017659033 | 53.65633254426 | 0.03757481815101 | {'exc': 15, 'op': 15} |
| 11_P536_UNIT_MAXMIN | 10 | 1.237491323504 | 46.12169334456 | 46.85098857008 | 56.16910536076 | 53.46082540756 | 0.03756511439405 | {'exc': 15, 'op': 15} |
| 11_P536_Nash-MTL | 0 | 1.237333214794 | 46.1191207968 | 47.09193050568 | 56.46606900362 | 53.77353949162 | 0.03757695455659 | None |
| 11_P536_Nash-MTL | 5 | 1.237382980036 | 46.1212042283 | 46.94160330613 | 56.27687881233 | 53.60120599137 | 0.03757515511838 | {'exc': 15, 'op': 13} |
| 11_P536_Nash-MTL | 10 | 1.237558535626 | 46.12404423078 | 46.78146339458 | 56.0825344317 | 53.37514069479 | 0.03756309877035 | {'exc': 15, 'op': 15} |
| 23_P67_UNIT_MAXMIN | 0 | 1.244320646101 | 46.36096468042 | 43.89745713223 | 52.94737764089 | 47.68686356456 | 0.03804893960762 | None |
| 23_P67_UNIT_MAXMIN | 5 | 1.244350888159 | 46.36374089466 | 43.60896644732 | 52.58109453362 | 47.36270264598 | 0.03804248198534 | {'exc': 15, 'op': 15} |
| 23_P67_UNIT_MAXMIN | 10 | 1.244456314092 | 46.36457937772 | 43.30268531359 | 52.19682936739 | 46.99216595308 | 0.03803065368701 | {'exc': 15, 'op': 15} |
| 23_P67_Nash-MTL | 0 | 1.244320646101 | 46.36096468042 | 43.89745713223 | 52.94737764089 | 47.68686356456 | 0.03804893960762 | None |
| 23_P67_Nash-MTL | 5 | 1.244388169516 | 46.36476277261 | 43.55465406462 | 52.51313049047 | 47.29449450395 | 0.03804068820247 | {'exc': 15, 'op': 15} |
| 23_P67_Nash-MTL | 10 | 1.244539681333 | 46.36708207978 | 43.19951230015 | 52.06793752272 | 46.86436547831 | 0.03802710860122 | {'exc': 15, 'op': 15} |
| 23_P536_UNIT_MAXMIN | 0 | 1.250805907035 | 46.64645772023 | 47.12993298885 | 56.82142954013 | 52.2250241953 | 0.03686370940187 | None |
| 23_P536_UNIT_MAXMIN | 5 | 1.250747445575 | 46.64390215453 | 46.91463470875 | 56.549624472 | 51.98157224735 | 0.03685761964684 | {'exc': 15, 'op': 15} |
| 23_P536_UNIT_MAXMIN | 10 | 1.250809988153 | 46.64344379873 | 46.57312441134 | 56.12425567442 | 51.56414139803 | 0.03684437799349 | {'exc': 15, 'op': 15} |
| 23_P536_Nash-MTL | 0 | 1.250805907035 | 46.64645772023 | 47.12993298885 | 56.82142954013 | 52.2250241953 | 0.03686370940187 | None |
| 23_P536_Nash-MTL | 5 | 1.250831917955 | 46.64703423557 | 46.81944225945 | 56.43039425606 | 51.86758674502 | 0.03685564556929 | {'exc': 15, 'op': 15} |
| 23_P536_Nash-MTL | 10 | 1.250953701394 | 46.64842024932 | 46.42671532193 | 55.94135043945 | 51.3861234782 | 0.03684049837198 | {'exc': 15, 'op': 15} |

Paired comparisons, UNIT_MAXMIN minus Nash:
| Start | Delta Rmax10 |
| --- | --- |
| 11_P67 | -9.2652026613349037e-05 |
| 11_P536 | -5.4320146753328302e-05 |
| 23_P67 | -6.6998197835177464e-05 |
| 23_P536 | -0.00011489651613905139 |

Method summary:
| Method | Trajectory passes | Scientific passes | Worst Rmax10 | Median Rmax10 | Paired wins | Min t | Median backtracks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNIT_MAXMIN | 4 | 0 | 1.0001277818361756 | 1.000067834618419 | 4 | 1.0 | 0.0 |
| Nash-MTL | 4 | 0 | 1.000182101982929 | 1.0001476597306433 | 0 | 1.0 | 0.0 |

Direction/solver stability:
| Arm | Min sampled p | Median sampled p | Gamma range | Active-face switches | Consecutive direction cosine range | Max solver residual | Near-zero progress labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | 0.08247208520157227 | 0.3262272102401203 | [0.08247208520157341, 0.759258799867313] | 8 | [0.15670929795931615, 0.8456543115954307] | 3.5388358909926865e-16 | 0 |
| 11_P67_Nash-MTL | 0.04530185632184558 | 0.21568899267619765 | None | None | [0.10984139185986454, 0.9338209574156607] | 4.974909373345326e-13 | 0 |
| 11_P536_UNIT_MAXMIN | 0.08646609717870563 | 0.26831065834665035 | [0.08646609717870715, 0.5342301539717044] | 7 | [-0.7075982372175721, 0.8717046653369984] | 2.5413698923060224e-16 | 0 |
| 11_P536_Nash-MTL | 0.04767112758887688 | 0.16524808681677808 | None | None | [-0.4188127846795443, 0.944411879770744] | 4.4431902601616e-11 | 0 |
| 23_P67_UNIT_MAXMIN | 0.07453456353458086 | 0.1538603642809051 | [0.0745345635345819, 0.9635176569774309] | 7 | [0.245226498303861, 0.8059567465926735] | 3.5735303605122226e-16 | 0 |
| 23_P67_Nash-MTL | 0.04153720102519142 | 0.08526218935939628 | None | None | [0.212314703111021, 0.8623029354525524] | 1.2290168882600483e-13 | 0 |
| 23_P536_UNIT_MAXMIN | 0.0741870469173914 | 0.15702247836096264 | [0.07418704691739413, 0.959486936720424] | 8 | [-0.0861309197214366, 0.918963323173396] | 2.7755575615628914e-16 | 0 |
| 23_P536_Nash-MTL | 0.041288084470950656 | 0.09408678181554722 | None | None | [-0.26434307021963976, 0.9379290656673502] | 7.061962126186927e-11 | 0 |

All arms accepted eta/eta0=1 throughout. No severe or systematic tiny-step/backtracking pathology. Active-face switches and consecutive direction changes alone are not evidence of instability: strict sampled descent and full displacement acceptance persisted. Nash alpha and iteration trajectories, native norms and raw task-norm ranges are retained in compact results; all raw alpha trajectories remain arm-local.

UNIT_MAXMIN active faces switched 7–8 times over nine transitions. Consecutive directions sometimes reversed, including cosine −0.7076 for seed11/P536; Nash also showed reversals. These changes track different frozen samples and did not produce a solver failure, tiny accepted step, or backtracking. Chemistry sets Rmax in all eight endpoints. Nash achieved larger AE/Exc/operator reductions in every paired endpoint, while UNIT_MAXMIN limited chemistry regression more effectively. The winner reflects the predeclared worst-objective rule, not dominance over all individual endpoints. Four paired starts provide descriptive evidence only, not a statistical generalization.

Read-only manifest-entry10 stress probes:
| Arm | Solver status | p_min | Read-only |
| --- | --- | --- | --- |
| 11_P67_UNIT_MAXMIN | PASS | 0.5022097378691207 | True |
| 11_P67_Nash-MTL | PASS | 0.37138746826799757 | True |
| 11_P536_UNIT_MAXMIN | PASS | 0.2282602203553853 | True |
| 11_P536_Nash-MTL | PASS | 0.13969347051189004 | True |
| 23_P67_UNIT_MAXMIN | PASS | 0.08913288996945813 | True |
| 23_P67_Nash-MTL | PASS | 0.05045042463377477 | True |
| 23_P536_UNIT_MAXMIN | PASS | 0.08826008428395425 | True |
| 23_P536_Nash-MTL | PASS | 0.05192379444227388 | True |

Full per-update gradient geometry, solver coefficients/residuals, trial losses/margins, hashes, full human metrics, all15 system ratios, stress probes and direction stability are preserved in the SHA-bound external arm result JSON files. Compact machine-readable results retain their exact paths/hashes and endpoint values.

Instrumentation caveats: the legacy Nash `efficiency` diagnostic uses denominator one and is excluded from interpretation and selection. Planned cursor5 checkpoint reload checks exact model/aggregator/cursor/RNG identity. This does not qualify crash recovery between sequential checkpoint/progress/monitor writes; interrupted runs require receipt reconciliation, not silent resume or restart.

Validation: {"Windows_tests": {"passed": 143, "skipped": 4}, "Ruff": "PASS", "compileall": "PASS", "git_diff_check": "PASS", "pre_execution_independent_review": "PASS", "production_sources_changed": 0, "post_run_tests": {"passed": 143, "skipped": 4}, "receipt_validation": {"status": "PASS", "accepted_states_reconstructed": 80, "immutable_hashes": 54, "mismatches": 0}}. Independent review: {"status": "PASS", "pre_execution": "PASS", "final_review": "PASS", "protocol_sha256": "65c0336fd17a9f5b3dab5c76d794cc0b7726253b4a6237897114476a7e56a743", "review_scope": "Read-only artifact review. No model forward, objective evaluation, gradient computation, optimizer step, or trajectory rerun.", "independent_update0_reconstruction": {"arms": ["11_P536_UNIT_MAXMIN", "11_P536_Nash-MTL", "23_P67_UNIT_MAXMIN", "23_P67_Nash-MTL"], "trainable_parameters": 9446, "named_parameter_blocks": 34, "direction_norms": "All saved candidate unit vectors independently recomputed with norm 1 within float64 roundoff; saved raw@unit dots, task p values, and slopes match diagnostics.", "armijo_slopes_max_abs_discrepancy": 1.1102230246251565e-16, "accepted_step": "t=1, zero backtracks on each checked update; all actual task losses strictly decrease and Armijo margins are positive.", "accepted_f32_state_hashes": "Reconstructed checkpoint0 + eta0 * (-saved unit direction), cast to stored F32 tensors, matched each of the four update0 after_sha256 receipts exactly.", "maxmin_faces": {"11_P536": ["relchem", "op"], "23_P67": ["relchem", "op"]}, "maxmin_gamma": {"11_P536": 0.31255714866612416, "23_P67": 0.7815416169917504}, "nash_solver": {"11_P536": {"status": "converged", "iterations": 19, "residual": 1.1102230246251565e-16}, "23_P67": {"status": "converged", "iterations": 19, "residual": 3.3306690738754696e-16}}, "fair_step_budget": "Within each start, both methods use the same frozen historical eta0; the candidate direction is globally normalized before the controller applies -eta0*d."}, "receipt_and_provenance_checks": {"all_eight_arms": "TRAJECTORY-PASS; cursor 10; 10/10 accepted; t=1 and zero backtracks on all 80 updates; exact cursor-5 resume receipts.", "unit_direction_norms": "joint_gradient_norm=1 on all 80 logged updates.", "manifest": "All arm sample entries 0..9 match the byte-hash-pinned manifest and match across arms at each cursor; entry 10 is reserved for the read-only stress probe.", "artifact_hashes": "Arm geometry, checkpoints, monitor records, and baselines match their recorded SHA256 values.", "immutable_inputs": {"protocol_immutable_files_checked": 53, "mismatches": 0, "receipt_validator_additional_hash_check_count": 54, "production_sources_changed": 0}, "receipt_validation_artifact": "PASS; records exact reconstruction of all 80 accepted after-state hashes and all cursor-5 reload checks. This reviewer independently reconstructed the four designated update0 hashes above.", "stress_probes": "All 8 are recorded PASS/read_only with positive minimum sampled directional progress.", "receipt_validator_hash_checks": 54}, "scientific_interpretation": {"frozen_relative_classifier": "CLEAR UNIT_MAXMIN WIN is correct under the literal frozen promotion rule: both methods have 4 trajectory passes and 0 scientific passes, UNIT_MAXMIN has 4/4 paired lower Rmax endpoints and lower finite worst Rmax (1.0001277818361756 vs 1.000182101982929), with no systematic pathology.", "absolute_outcome": "Both methods have zero scientific passes. Chemistry is the Rmax-setting objective at all 8 final endpoints, and every final chemistry ratio is greater than 1; neither method improves the complete four-objective baseline gate.", "tradeoff": "UNIT_MAXMIN limits chemistry regression and therefore wins the frozen worst-objective comparison; Nash has lower (better) AE17, Exc, and operator ratios in all four paired final endpoints. This is not across-objective dominance.", "inference_limit": "Four paired starts on one fixed manifest support descriptive comparison only; paired wins are not a significance test or a generalization claim. The result does not qualify a production trainer or establish initialization superiority."}, "report_review": {"status": "PASS", "science_nonqualification_explicit": true, "MOO_tradeoff_and_descriptive_limit_explicit": true, "manifest_label_discrepancy_disclosed": true, "legacy_nash_efficiency_excluded_from_interpretation": true, "cursor5_reload_vs_crash_recovery_distinction_explicit": true, "followup_not_launched": true}, "validation_reviewed": {"Windows_tests_passed": 143, "Windows_tests_skipped": 4, "Ruff": "PASS", "compileall": "PASS", "git_diff_check": "PASS"}, "findings": [], "nonblocking_caveat": "The frozen label CLEAR UNIT_MAXMIN WIN is relative to the stated classifier only; its zero scientific-pass count and chemistry regressions are decisive limits on interpretation."}. Immutable hash mismatches: zero. No initialization is declared superior.

Next experiment: Preregistered matched 25-update qualification of UNIT_MAXMIN on the same four starts and manifest/controller. Not launched.

Exactly eight candidate trajectories were attempted; no third MOO method or new initialization was introduced; no arm was pruned competitively at t5; no 25/100-update extension, full90, SCF, Diet, or unrelated Slurm sweep was launched.
