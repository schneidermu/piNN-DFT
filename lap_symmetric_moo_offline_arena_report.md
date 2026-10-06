# Symmetric MOO offline direction Arena

Starting commit: `3d127128cf2168aedfe3f9419eee1c04261e0dc7`.

**Selected: UNIT_MAXMIN and Nash-MTL.** Both have strict common descent in all four saved states. This is a local direction screen, not evidence of trajectory superiority. No objective is scientifically secondary. P67 and P536 remain factors; neither is privileged.

## Frozen scope and provenance

Only seed11/23 P67/P536 saved gradient arrays are used. Canonical task order: full251 chemistry, ae17, exc equal15mean, op equal15mean. Each aggregate has9446 F64 stored coordinates (chemistry F64 shadow; Exc/operator gradients produced by their existing production precision paths then widened). Exact ordering is proven by SHA-bound executed matrix.py plus sorted unique named_trainable_parameters. Tied exchange coordinates occur once. AE17 is a separate fourth-task vector; not folded into chemistry.

| State | Tensor-state SHA | State-file SHA | Gradient artifact SHA (contains all four labeled aggregates) |
|---|---|---|---|
| 11_P67 | ee815caf918e8ded13e99896070a95a32293100499746fd0623444e2c0023523 | 264a8d189d53d49007d7de11d8cfb5382e8b8262e4b48068142c8771e293c834 | d79c78b34e81a44f6a5ea1bebba1283bbf293c522e662f0c750b3cbcf425a381 |
| 11_P536 | 3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6 | 0ca0f77367214c171e6edf0576bdb043d64479245ad1a0e539fa11fb0cee542d | e7c4c07f53dc4b1598c29d29bb1e51a1704c22474c29f1ecd3070bdfc611a7e5 |
| 23_P67 | 69f6e7c90a5f4cf6f51e885b8f3c1b9679508c71a57a8363149de6783820e4d8 | 1cac6eb351372adab090c60f7daaf0013935857ed56648ccf0cd68b541b62666 | 173703ef9c7d63fddfff58fabc31efbe96223b23dbb407ea38ea05473fcbe625 |
| 23_P536 | 0b0c2aefc703c90be250c2fdb79e60be1df35273c7bc7d680ede50a8e7145074 | ca1d4df5cb84b66e17f9964d645de5d4f00fa9f5f3afede56fa632b5ff7057a7 | cf1a6fd2f39a0788646a7f7a36f22ad6c4c91c21f31ba7a261832611f3917981 |

Frozen protocol SHA: `5c36162fa57bbc2f61a8e0298ba1ae0ca6c94b16fe2f5ba941432fa1e6ba1fe4`. All46 immutable hashes match before/after, including objective-source/order evidence, state/gradient receipts, four existing exact Jacobians and production aggregator source. No state/model is loaded into a functional, mutated, differentiated or updated; only saved gradients/Jacobians are read. No new large gradient copies need persistence.

## Candidate contracts

Existing repository aggregate_task_gradients is used unchanged for RAW_EQUAL_MEAN, IMTL-G, CAGrad, Nash-MTL, PCD_CONTROL with explicit task_order and state=None independently per candidate/state. Equal fixed weights, CAGrad c=.4/paper_unscaled/max_iter500/ftol1e-12, Nash max_iter100/tol1e-10, PCD tau=.02/beta=.999/eps1e-8/qp_tolerance1e-9. UNIT_MEAN and UNIT_MAXMIN are diagnostic-only. No production aggregator or optimizer edit.

UNIT_MAXMIN dual: minimize ||sum w_i u_i|| over simplex w. Let z be this minimum-norm convex combination. Projection KKT implies u_i dot z>=||z||Р вЂ™Р вЂ . If ||z||>0, d=z/||z|| has min_i u_i dot d=||z||. For every unit-ball d, min_i u_i dot d<=z dot d<=||z||; hence gamma*=||z|| and this direction solves the stated max-min problem. If z=0, gamma*=0 since d=0 is allowed. The reused exhaustive4-task simplex face solver is SHA-bound; primal/active/dual/complementarity certificate<=1e-9. No raw-gradient MGDA substitution.

All primary metrics use d_hat=d/||d||. Gradient-space convention: model update would be -alpha*d, so positive p predicts descent. Native norm is diagnostic only. For UNIT_MEAN the recorded native_norm is the pre-normalization mean; the evaluated direction is its unit-normalized version as required. Near-zero label tolerance1e-12 does not alter any raw p, tier or selection; strict common descent uses raw p>0.

## Task norms and optimal margin

| State | Chemistry norm | AE17 norm | Exc norm | Operator norm | gamma_star |
|---|---:|---:|---:|---:|---:|
| 11_P67 | 10.941295652169918 | 5540.5772111629776 | 15019.189669219606 | 1.0791263393671255 | 0.24537016305600881 |
| 11_P536 | 9.6245322379200502 | 5545.3854445749676 | 15023.374357813658 | 0.62964309078250558 | 0.14564137649058295 |
| 23_P67 | 12.874764904114027 | 11077.62326418372 | 29791.639767765872 | 0.76745024006948048 | 0.17873404416979743 |
| 23_P536 | 15.384958717373509 | 11005.969625584356 | 29512.896890871718 | 0.7155310851415555 | 0.18276706524730937 |

## Full candidate progress

| State | Candidate | p_chem | p_AE | p_Exc | p_op | p_min | p_mean | Range | Std | Efficiency | Native norm | Strict descent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11_P67 | UNIT_MAXMIN | 0.2453701630560067 | 0.24537016305601081 | 0.24742799761113451 | 0.24537016305600942 | 0.2453701630560067 | 0.24588462169479036 | 0.0020578345551278132 | 0.00089106850076208204 | 0.99999999999999145 | 1 | True |
| 11_P67 | UNIT_MEAN | -0.59430579797401872 | 0.84012962238385835 | 0.85343824222550035 | 0.85140413341382304 | -0.59430579797401872 | 0.48766655001229076 | 1.447744040199519 | 0.6246975947600395 | -2.4220785060910646 | 0.48766655001229092 | False |
| 11_P67 | RAW_EQUAL_MEAN | -0.87853480798423933 | 0.99975003360060466 | 0.99996673630766431 | 0.53694583436458709 | -0.87853480798423933 | 0.41453194907215418 | 1.8785015442919035 | 0.77010078789749381 | -3.5804467708802181 | 5137.2123629650059 | False |
| 11_P67 | IMTL-G | 0.23935138618209481 | 0.23935138617685009 | 0.23935138617618593 | 0.2393513861772382 | 0.23935138617618593 | 0.23935138617809223 | 5.9088844928112394e-12 | 2.3413102374941019e-12 | 0.97547062444406074 | 1.0522719342606848 | True |
| 11_P67 | CAGrad | -0.7077398480018704 | 0.95825273657780985 | 0.95938389765763255 | 0.5566041600141306 | -0.7077398480018704 | 0.44162523656192565 | 1.6671237456595029 | 0.68360044166253486 | -2.8843761571789801 | 3473.5008048878767 | False |
| 11_P67 | Nash-MTL | 0.14505902397934253 | 0.30607897737930584 | 0.3189322143331314 | 0.58509996935220854 | 0.14505902397934253 | 0.33879254626099709 | 0.44004094537286598 | 0.15784853079673528 | 0.59118444627772859 | 2 | True |
| 11_P67 | PCD_CONTROL | 0.4383864515623499 | 0.042001768954779979 | 0.044000297046703017 | 0.1370372386445958 | 0.042001768954779979 | 0.16535643905210717 | 0.39638468260756993 | 0.16224291408280389 | 0.17117716527413543 | 10.941295652169922 | True |
| 11_P536 | UNIT_MAXMIN | 0.1456413764905844 | 0.14564137649058034 | 0.14677838310185673 | 0.1456413764905822 | 0.14564137649058034 | 0.1459256281434009 | 0.0011370066112763866 | 0.00049233830481725113 | 0.99999999999998213 | 0.99999999999999989 | True |
| 11_P536 | UNIT_MEAN | -0.81929552670129968 | 0.97793956514076452 | 0.98125379377208488 | 0.76847461236372416 | -0.81929552670129968 | 0.47709311114381847 | 1.8005493204733845 | 0.75341752102944237 | -5.6254310858856424 | 0.47709311114381819 | False |
| 11_P536 | RAW_EQUAL_MEAN | -0.82558829128510813 | 0.99979341025408042 | 0.99996977802059595 | 0.69685931582289151 | -0.82558829128510813 | 0.46775855320311494 | 1.8255580693057041 | 0.75689218131111169 | -5.6686383442585075 | 5139.9132545092289 | False |
| 11_P536 | IMTL-G | 0.14372941233513745 | 0.14372941230352254 | 0.14372941230309666 | 0.14372941230820677 | 0.14372941230309666 | 0.14372941231249087 | 3.2040786690501477e-11 | 1.3227840648140564e-11 | 0.98687210850681628 | 0.1887505601856829 | True |
| 11_P536 | CAGrad | -0.60215619483477789 | 0.95279831304846929 | 0.9455228296215521 | 0.46624561015066807 | -0.60215619483477789 | 0.44060263949647788 | 1.5549545078832472 | 0.63350066198094268 | -4.1345132087083289 | 3632.7566138225493 | False |
| 11_P536 | Nash-MTL | 0.079712665578949488 | 0.32119707719261087 | 0.31727692415619291 | 0.12984920103997846 | 0.079712665578949488 | 0.21200896699193292 | 0.24148441161366138 | 0.10869213946229825 | 0.5473215613565946 | 2.0000000000000018 | True |
| 11_P536 | PCD_CONTROL | 0.23012975455848558 | 0.066589708262828895 | 0.066818076121126391 | 0.066589707423004851 | 0.066589707423004851 | 0.10753181159136142 | 0.16354004713548073 | 0.070782016774049494 | 0.45721696009451329 | 9.6245322379200502 | True |
| 23_P67 | UNIT_MAXMIN | 0.1787340441697986 | 0.18410183031381308 | 0.17873404416979674 | 0.17873404416979746 | 0.17873404416979674 | 0.18007599070580146 | 0.0053677861440163399 | 0.0023243195813997604 | 0.99999999999999611 | 1.0000000000000002 | True |
| 23_P67 | UNIT_MEAN | -0.85403025215817285 | 0.97761795061354495 | 0.97736224126408322 | 0.97599360974293514 | -0.85403025215817285 | 0.51923588736559756 | 1.8316482027717178 | 0.79285581585771614 | -4.77821814039436 | 0.51923588736559789 | False |
| 23_P67 | RAW_EQUAL_MEAN | -0.93552223805579371 | 0.99998159331897774 | 0.99999722893448506 | 0.96566857353920854 | -0.93552223805579371 | 0.50753128943421943 | 1.9355194669902787 | 0.83326515252537359 | -5.2341580609402376 | 10214.418262196783 | False |
| 23_P67 | IMTL-G | -0.088546143686937848 | -0.088546143571266764 | -0.088546143572143021 | -0.088546143574641148 | -0.088546143686937848 | -0.088546143601247199 | 1.1567108382237734e-10 | 4.9489011094296063e-11 | -0.49540726333489643 | 0.13420587214469823 | False |
| 23_P67 | CAGrad | -0.83473613254350942 | 0.97629436243442436 | 0.97508093937082485 | 0.94184861068210479 | -0.83473613254350942 | 0.51462194498596114 | 1.8110304949779339 | 0.77917484427734529 | -4.6702693738105632 | 6553.0266571088923 | False |
| 23_P67 | Nash-MTL | 0.093834836086255147 | 0.25867097028497349 | 0.25458671889713441 | 0.3018355353585187 | 0.093834836086255147 | 0.22723201515672042 | 0.20800069927226356 | 0.079210448058857796 | 0.52499699496035579 | 2.0000000000000031 | True |
| 23_P67 | PCD_CONTROL | 0.29804641229686923 | 0.062305126034588876 | 0.056766499949709018 | 0.056955481384757145 | 0.056766499949709018 | 0.11851837991648106 | 0.24127991234716023 | 0.10367440556547905 | 0.3176031752282224 | 12.874764904114029 | True |
| 23_P536 | UNIT_MAXMIN | 0.18276706524730807 | 0.1837872266807126 | 0.18276706524731062 | 0.21514746938837473 | 0.18276706524730807 | 0.19111720664092649 | 0.03238040414106666 | 0.013880128390214306 | 0.99999999999999289 | 1.0000000000000002 | True |
| 23_P536 | UNIT_MEAN | -0.83930824602353793 | 0.97259808297070482 | 0.97334896379145863 | 0.97809673680176579 | -0.83930824602353793 | 0.52118388438509777 | 1.8174049828253036 | 0.78548332721430347 | -4.5922291573037874 | 0.52118388438509766 | False |
| 23_P536 | RAW_EQUAL_MEAN | -0.93303898385458572 | 0.99998334170771153 | 0.99999763273021747 | 0.96187659475932075 | -0.93303898385458572 | 0.50720464633566598 | 1.9330366165848032 | 0.83167061764847638 | -5.1050717622020887 | 10126.236699290863 | False |
| 23_P536 | IMTL-G | -0.14959487378125144 | -0.14959487350722503 | -0.14959487350542408 | -0.14959487351029471 | -0.14959487378125144 | -0.14959487357604884 | 2.7582736095155269e-10 | 1.1848658849609647e-10 | -0.81850016893814359 | 0.12254318586591127 | False |
| 23_P536 | CAGrad | -0.82886744363264175 | 0.97477037098827735 | 0.97454103302805029 | 0.94601293304840484 | -0.82886744363264175 | 0.51661422335802265 | 1.8036378146209191 | 0.77690221178693186 | -4.5351028781420126 | 6512.1135616146821 | False |
| 23_P536 | Nash-MTL | 0.098063442060733003 | 0.2590528308037322 | 0.2594749719871916 | 0.33886339780166375 | 0.098063442060733003 | 0.23886366066333015 | 0.24079995574093074 | 0.087545813337535167 | 0.53654875908873112 | 2.0000000000000018 | True |
| 23_P536 | PCD_CONTROL | 0.30696829712518114 | 0.056605995657253751 | 0.055565858254441047 | 0.093339945902911461 | 0.055565858254441047 | 0.12812002423494687 | 0.2514024388707401 | 0.10437279560443435 | 0.30402555394350006 | 15.384958717373507 | True |

Raw derivatives g_i dot d_hat and all Gram matrices are retained in JSON, without clipping efficiency. No solver failures were replaced by fallback directions.

## Cross-state robustness

| Candidate | Tier | Common descent states | Worst p_min | Median p_min | Min efficiency | Median efficiency | Efficiency range | Median imbalance | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNIT_MAXMIN | 1 | 4 | 0.14564137649058034 | 0.1807505547085524 | 0.9999999999999821 | 0.9999999999999922 | 1.3988810110276972e-14 | 0.0037128103495720766 | 0 |
| UNIT_MEAN | 3 | 0 | -0.8540302521581729 | -0.8293018863624189 | -5.6254310858856424 | -4.685223648849074 | 3.203352579794578 | 1.808977151649344 | 0 |
| RAW_EQUAL_MEAN | 3 | 0 | -0.9355222380557937 | -0.9057868959194125 | -5.6686383442585075 | -5.169614911571163 | 2.0881915733782894 | 1.9057690804383534 | 0 |
| IMTL-G | 3 | 2 | -0.14959487378125144 | 0.027591634308079406 | -0.8185001689381436 | 0.24003168055458216 | 1.8053722774449599 | 7.385593525643941e-11 | 0 |
| CAGrad | 3 | 0 | -0.8347361325435094 | -0.7683036458172561 | -4.670269373810563 | -4.334808043425171 | 1.785893216631583 | 1.735380780140211 | 0 |
| Nash-MTL | 1 | 4 | 0.07971266557894949 | 0.09594913907349407 | 0.5249969949603558 | 0.5419351602226629 | 0.0661874513173728 | 0.24114218367729606 | 0 |
| PCD_CONTROL | 1 | 4 | 0.04200176895477998 | 0.05616617910207503 | 0.17117716527413543 | 0.3108143645858612 | 0.2860397948203779 | 0.24634117560895016 | 0 |

Tier1=alltasks strictdescent in all4 states; Tier2=negative progress in at most1 state and every p_min>=-.02; otherwiseTier3. RAW_EQUAL_MEAN and PCD_CONTROL are ineligible controls. Slot1 reserved UNIT_MAXMIN. Among IMTL-G/CAGrad/Nash-MTL only Nash-MTL is Tier1/2. No UNIT_MEAN fallback was needed.

Exact lexicographic selection uses tier, worst p_min, minimum efficiency, smaller efficiency range, median p_min, smaller median imbalance. Exact ties use simplicity IMTL-G then Nash-MTL then CAGrad. No practical tie tolerance was introduced after observing results.

## Coefficients and solver diagnostics

| State | Method | Raw-gradient coefficients in task order | Negative tasks | Solver status | Residual |
|---|---|---|---|---|---:|
| 11_P67 | UNIT_MAXMIN | [0.1860399344713591, 0.00036661495840864777, 0.0, 0.008060901850400565] | [] | exact simplex / unit-gradient dual | 5.06539255e-16 |
| 11_P67 | UNIT_MEAN | [0.022849213470474045, 4.512165257733581e-05, 1.66453720544159e-05, 0.23166888887784662] | [] | closed form | 0 |
| 11_P67 | RAW_EQUAL_MEAN | {'full251': 0.25, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | closed_form | 0 |
| 11_P67 | IMTL-G | {'full251': 0.18152103383447332, 'ae17': 0.004860867474369667, 'exc': -0.0016912737275099944, 'op': 0.8153093724186677} | ['exc'] | converged | 6.66133815e-16 |
| 11_P67 | CAGrad | {'full251': 188.0600190792734, 'ae17': 0.25, 'exc': 0.25, 'op': 0.2500000000022728} | [] | converged | 4.4408921e-16 |
| 11_P67 | Nash-MTL | {'full251': 0.31503332703697273, 'ae17': 0.00029483666577609586, 'exc': 0.00010438187995038631, 'op': 0.7918950641352355} | [] | converged | 3.21964677e-15 |
| 11_P67 | PCD_CONTROL | {'full251': 2.1000884476513284, 'ae17': 0.003730708583478027, 'exc': 0.0, 'op': 0.0} | [] | active_set | 0 |
| 11_P536 | UNIT_MAXMIN | [0.3424522362283871, 0.00022605536935806157, 0.0, 3.6793445905056896] | [] | exact simplex / unit-gradient dual | 4.4408921e-16 |
| 11_P536 | UNIT_MEAN | [0.025975288338171466, 4.5082528978138784e-05, 1.664073556617292e-05, 0.39705033480047547] | [] | closed form | 0 |
| 11_P536 | RAW_EQUAL_MEAN | {'full251': 0.25, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | closed_form | 0 |
| 11_P536 | IMTL-G | {'full251': 0.06297693951240102, 'ae17': 0.0008124001703722367, 'exc': -0.00029205307463094164, 'op': 0.9365027133918574} | ['exc'] | converged | 2.22044605e-16 |
| 11_P536 | CAGrad | {'full251': 213.86716611050647, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | converged | 0 |
| 11_P536 | Nash-MTL | {'full251': 0.6517229890510906, 'ae17': 0.00028071568628318735, 'exc': 0.00010489723203431462, 'op': 6.115560690715863} | [] | converged | 5.66213743e-15 |
| 11_P536 | PCD_CONTROL | {'full251': 3.3294854129617204, 'ae17': 0.0021147497425602034, 'exc': 0.0, 'op': 35.040751182060255} | [] | active_set | 0 |
| 23_P67 | UNIT_MAXMIN | [0.21720227372512702, 0.0, 9.193195837431068e-05, 0.07775490002060266] | [] | exact simplex / unit-gradient dual | 4.4408921e-16 |
| 23_P67 | UNIT_MEAN | [0.019417830295302287, 2.2568017889568646e-05, 8.391615968399847e-06, 0.32575401888905064] | [] | closed form | 0 |
| 23_P67 | RAW_EQUAL_MEAN | {'full251': 0.25, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | closed_form | 0 |
| 23_P67 | IMTL-G | {'full251': -0.06148917279308186, 'ae17': 0.003437698888243457, 'exc': -0.001329854871078381, 'op': 1.0593813287759168} | ['full251', 'exc'] | converged | 4.06667294e-16 |
| 23_P67 | CAGrad | {'full251': 317.59694460891785, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | converged | 0 |
| 23_P67 | Nash-MTL | {'full251': 0.4138725254968872, 'ae17': 0.00017449208053540603, 'exc': 6.592343862046055e-05, 'op': 2.158486862735506} | [] | converged | 4.21884749e-15 |
| 23_P67 | PCD_CONTROL | {'full251': 2.838324997399816, 'ae17': 0.0, 'exc': 0.0011727547592476082, 'op': 0.0} | [] | active_set | 0 |
| 23_P536 | UNIT_MAXMIN | [0.17781800392624392, 0.0, 9.26958359840703e-05, 0.0] | [] | exact simplex / unit-gradient dual | 1.11022302e-16 |
| 23_P536 | UNIT_MEAN | [0.01624963736286707, 2.2714945480028653e-05, 8.470872951727233e-06, 0.349390830379566] | [] | closed form | 0 |
| 23_P536 | RAW_EQUAL_MEAN | {'full251': 0.25, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | closed_form | 0 |
| 23_P536 | IMTL-G | {'full251': -0.027881193639829827, 'ae17': 0.0022829136386270234, 'exc': -0.0008894544695063568, 'op': 1.026487734470709} | ['full251', 'exc'] | converged | 5.56192326e-16 |
| 23_P536 | CAGrad | {'full251': 263.52627874245206, 'ae17': 0.25, 'exc': 0.25, 'op': 0.25} | [] | converged | 0 |
| 23_P536 | Nash-MTL | {'full251': 0.33141070762748176, 'ae17': 0.00017536921260079475, 'exc': 6.52924086423672e-05, 'op': 2.0621337839743057} | [] | converged | 9.32587341e-15 |
| 23_P536 | PCD_CONTROL | {'full251': 2.778292912663347, 'ae17': 0.0, 'exc': 0.0013805212518587257, 'op': 0.0} | [] | active_set | 0 |

JSON records coefficient signs, absolute normalized fractions, entropy/concentration, and abs(coefficient)*gradient-norm contribution fractions. These are labeled differently: raw coefficients alone do not describe task contribution under unequal norms. UNIT_MAXMIN unit-gradient simplex weights w_i and equivalent raw coefficients c_i=w_i/(||g_i|| gamma_star) for the returned unit direction are both recorded; omitting gamma_star instead reconstructs the unnormalized dual vector z, not d_hat. CAGrad simplex weights are recovered from its unchanged final coefficients: normalize (coefficient-.25); c>0 ensures positive total excess. Nash positive alpha and residuals are recorded. PCD active constraints, bias-corrected EMA scales/state and all other repository diagnostics are retained. PCD EMA is cold-started for each state and never written to training state.

## Initialization robustness

| Seed | Method | P536-P67 p_min | Efficiency change | Imbalance change |
|---|---|---:|---:|---:|
| 11 | UNIT_MAXMIN | -0.0997287866 | -9.32587341e-15 | -0.000920827944 |
| 11 | UNIT_MEAN | -0.224989729 | -3.20335258 | 0.35280528 |
| 11 | RAW_EQUAL_MEAN | 0.0529465167 | -2.08819157 | -0.052943475 |
| 11 | IMTL-G | -0.0956219739 | 0.0114014841 | 2.61319022e-11 |
| 11 | CAGrad | 0.105583653 | -1.25013705 | -0.112169238 |
| 11 | Nash-MTL | -0.0653463584 | -0.0438628849 | -0.198556534 |
| 11 | PCD_CONTROL | 0.0245879385 | 0.286039795 | -0.232844635 |
| 23 | UNIT_MAXMIN | 0.00403302108 | -3.21964677e-15 | 0.027012618 |
| 23 | UNIT_MEAN | 0.0147220061 | 0.185988983 | -0.0142432199 |
| 23 | RAW_EQUAL_MEAN | 0.0024832542 | 0.129086299 | -0.00248285041 |
| 23 | IMTL-G | -0.0610487301 | -0.323092906 | 1.60156277e-10 |
| 23 | CAGrad | 0.00586868891 | 0.135166496 | -0.00739268036 |
| 23 | Nash-MTL | 0.00422860597 | 0.0115517641 | 0.0327992565 |
| 23 | PCD_CONTROL | -0.0012006417 | -0.0135776213 | 0.0101225265 |

No cross-seed parameter-vector cosine or average direction is computed. Paired tables compare scalar metrics, not parameters at different function points.

## Selected-method diversity

| State | abs cosine Nash-MTL / UNIT_MAXMIN | Maximum progress-vector difference |
|---|---:|---:|
| 11_P67 | 0.922085391 | 0.339729806 |
| 11_P536 | 0.966187767 | 0.175555701 |
| 23_P67 | 0.97767689 | 0.123101491 |
| 23_P536 | 0.978125937 | 0.123715928 |

Redundancy requires all4 absolute cosines>=.995 AND all4 maximum progress difference<=.01. Nash-MTL is nonredundant; no replacement required.

## Secondary q diagnostics

| State | Candidate | ||-J_q d_hat|| | Cos chemistry response | Cos AE response | Cos Exc response | Cos operator response |
|---|---|---:|---:|---:|---:|---:|
| 11_P67 | UNIT_MAXMIN | 18.3972987 | -0.9838601 | 0.997159878 | 0.996554018 | 0.97734589 |
| 11_P67 | UNIT_MEAN | 39.211213 | -0.989807639 | 0.992374081 | 0.994526684 | 0.998195636 |
| 11_P67 | RAW_EQUAL_MEAN | 16.4727624 | -0.994949407 | 0.999897326 | 0.999988459 | 0.985835136 |
| 11_P67 | IMTL-G | 17.3846063 | -0.97835896 | 0.994409315 | 0.993673386 | 0.973999892 |
| 11_P67 | CAGrad | 20.4704437 | -0.992838554 | 0.999772654 | 0.999783515 | 0.985161113 |
| 11_P67 | Nash-MTL | 36.1379275 | -0.988424915 | 0.993744628 | 0.995570251 | 0.996684807 |
| 11_P67 | PCD_CONTROL | 15.3622011 | -0.980224538 | 0.995527644 | 0.994718988 | 0.974000644 |
| 11_P536 | UNIT_MAXMIN | 10.2724336 | -0.968344889 | 0.992672157 | 0.991784178 | 0.967484783 |
| 11_P536 | UNIT_MEAN | 20.3787469 | -0.992794447 | 0.997961122 | 0.998955297 | 0.991695844 |
| 11_P536 | RAW_EQUAL_MEAN | 14.9055834 | -0.985877797 | 0.999906777 | 0.999988277 | 0.984162315 |
| 11_P536 | IMTL-G | 8.40730079 | -0.964766494 | 0.987893466 | 0.987151473 | 0.964440212 |
| 11_P536 | CAGrad | 4.94139342 | -0.699392904 | 0.817147004 | 0.806359319 | 0.692499708 |
| 11_P536 | Nash-MTL | 5.44437857 | -0.791099473 | 0.883892627 | 0.875901905 | 0.787605885 |
| 11_P536 | PCD_CONTROL | 7.8579784 | -0.948568701 | 0.983768991 | 0.98178454 | 0.94756119 |
| 23_P67 | UNIT_MAXMIN | 13.2687151 | -0.926662755 | 0.988537973 | 0.987562007 | 0.848927979 |
| 23_P67 | UNIT_MEAN | 8.60743238 | -0.975749421 | 0.993444181 | 0.994276281 | 0.937437927 |
| 23_P67 | RAW_EQUAL_MEAN | 8.13905631 | -0.973779564 | 0.99997959 | 0.999997008 | 0.897295273 |
| 23_P67 | IMTL-G | 3.44554507 | 0.176523955 | 0.0174778528 | 0.00869145725 | -0.407916907 |
| 23_P67 | CAGrad | 10.6132948 | -0.962315675 | 0.999151549 | 0.998856754 | 0.884122466 |
| 23_P67 | Nash-MTL | 11.1108234 | -0.941969881 | 0.986763569 | 0.986985074 | 0.911618808 |
| 23_P67 | PCD_CONTROL | 12.6678265 | -0.920249881 | 0.98595535 | 0.984819287 | 0.839147755 |
| 23_P536 | UNIT_MAXMIN | 5.57618609 | -0.218296375 | 0.447316579 | 0.454483161 | 0.902963927 |
| 23_P536 | UNIT_MEAN | 6.24949394 | -0.83036165 | 0.939857944 | 0.942648472 | 0.941466324 |
| 23_P536 | RAW_EQUAL_MEAN | 8.0240499 | -0.969009525 | 0.999980906 | 0.999997525 | 0.774981139 |
| 23_P536 | IMTL-G | 4.51082529 | -0.088472715 | -0.147282817 | -0.155556572 | -0.741413584 |
| 23_P536 | CAGrad | 8.16099807 | -0.925939556 | 0.989708615 | 0.990825849 | 0.852583153 |
| 23_P536 | Nash-MTL | 6.47202812 | 0.260527923 | -0.0204137567 | -0.012259735 | 0.614154569 |
| 23_P536 | PCD_CONTROL | 5.27273892 | -0.0436923117 | 0.283460986 | 0.291150986 | 0.816868767 |

Dominant q-mode signed step coefficients are in JSON; stored SVD signs are arbitrary and not compared across states. These optional diagnostics did not enter tier/ranking/selection. q-space does not define all semilocal physics, and local objective opposition does not imply incompatible final physical targets.

## Validation and independent review

49 focused/existing aggregator tests PASS; Ruff and compileall PASS. UNIT_MAXMIN identical/orthogonal/opposing/known4-vector/scaling cases pass. Existing candidates are bitwise deterministic under identical cold-start fixtures. Selection boundaries and redundancy tests pass. Direct flattened Gram/dots are checked against actual structured repository diagnostics on all4 states at relative scaled tolerance1e-10.

Independent Luna MAX Ponytail/Pocock/MOO/numerical/scientific review: PASS. Reviewer independently reconstructed 11_P67 saved vectors and structured aggregators offline. Norm relative difference<=2.3e-16; Gram relative agreement about6e-16; maxmin gamma difference2e-16, progress difference1.3e-15, KKT<5e-16. Largest candidate discrepancy: IMTL-G norm2.5e-10 absolute, progress1.4e-11; no tier/selection effect. No reviewer model/objective work or edits. Review receipt SHA: `c25a5fced8cb8ccc5630d85f1b2ce042ac9e5a0a56873b40d057a53cb06db05f`. Final46 immutable hashes and git diff --check PASS; state mutation NONE. Production changes ZERO. Existing unrelated untracked planning artifacts preserved.

## Next experiment Р Р†Р вЂљРІР‚Сњ not launched

Predeclare 2seeds(11,23) Р вЂњРІР‚вЂќ2regimes(P67,P536) Р вЂњРІР‚вЂќ2directionrules(UNIT_MAXMIN,Nash-MTL)=8 short trajectories, initially10updates, with one shared frozen stochastic manifest and identical validated precision/objective/acceptance contracts. This offline screen does not authorize or execute them; trajectory behavior and production integration remain unqualified.

No model update was applied, no training trajectory was run, no production optimizer or aggregator behavior was changed, no scientific gradients were recomputed, no cursor was advanced, and no full90/SCF/Diet/Slurm/10/25/100-update run was launched.
