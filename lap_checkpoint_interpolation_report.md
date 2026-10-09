# IID t70 to J251 checkpoint interpolation

Three evaluation-only parameter states were formed by `theta(alpha) = (1-alpha) theta_IID70 + alpha theta_J251`. Scientific objectives were measured at those states. Endpoint metrics were reused from frozen receipts and were not linearly interpolated.

## A. Provenance

Source commit `66c442ec22c2c546d275d73cc24334169e600a18`. Dataset logical SHA256 `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`. Evaluation manifest SHA256 `132bd550161be6bc6761f8df1de8d3755c0b0debab53663208c3a9bbf56cb805`.
IID t70 file `04b8e549c17375988be576e2878881d13f55b217b42ec16a7e6101fdddd05443`, tensor `59ab4b98550805b13852e13745281fb1d072efcbed5c2c8622bf3a6840c3ff51`.
J251 file `11cee17f61c017e26902c905b6d9ace0b576ab7eaacf5d5f93f5172a49e3257a`, tensor `d1b1372997199331b4b332b7f98dfa4e791617ea9a453d4e9cdb999c16aafff5`.
Shared P536 tensor `3c2e45d6a86cccf80d7b4dc458e67f9b406935ed60d4e57433bb441a6df88da6`. Unique trainable parameters: 9446. Interpolation arithmetic: F64, then one cast to F32.
New GPU seconds: 973.542. Peak allocated CUDA bytes: 14070156800. Peak reserved bytes: 14210301952.
No optimizer state was saved or stepped. Full30 was produced by the unchanged validation helper and then discarded. It did not enter selection.

## B. Five-point comparison

| Alpha | Clean28 | relchem/t0 | AE17/t0 | Exc/t0 | Op/t0 | Eligible |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 IID t70 | 8.629660 | 1.044608 | 0.258234 | 0.196700 | 0.941692 | No |
| 0.25 | 8.807149 | 1.024954 | 0.214448 | 0.137596 | 0.930864 | No |
| 0.50 | 8.988741 | 1.005609 | 0.169997 | 0.082907 | 0.926765 | No |
| 0.75 | 9.168651 | 0.991430 | 0.122273 | 0.054614 | 0.929238 | Yes |
| 1.00 J251 | 9.346761 | 0.979708 | 0.081012 | 0.101105 | 0.938121 | Yes |

Classification: **GO**.

Best eligible interpolated candidate: alpha 0.75, Clean28 9.168651.
Lowest Clean28 among the three new candidates: alpha 0.25, Clean28 8.807149, eligible False.

## C. Relchem databases

| Database | P536 mean | IID t70 mean | alpha 0.25 | alpha 0.50 | alpha 0.75 | J251 mean |
|---|---:|---:|---:|---:|---:|---:|
| ABDE4 | 5.610455 | 9.326408 | 8.291097 | 7.274566 | 6.662235 | 6.156657 |
| DBH76 | 1.011435 | 0.930999 | 0.935889 | 0.940965 | 0.946231 | 0.951690 |
| EA13 | 1.714299 | 1.870611 | 1.826569 | 1.783698 | 1.742078 | 1.701774 |
| IP13 | 2.361088 | 2.580558 | 2.585097 | 2.587555 | 2.587927 | 2.586206 |
| MGAE109 | 0.282011 | 0.207811 | 0.219538 | 0.231533 | 0.243433 | 0.255076 |
| NCCE31 | 2.617567 | 2.104050 | 2.137519 | 2.170683 | 2.203564 | 2.236182 |
| PA8 | 1.625910 | 2.552881 | 2.337074 | 2.120896 | 1.904396 | 1.733028 |
| pTC13 | 3.975477 | 5.106771 | 4.929791 | 4.753236 | 4.577138 | 4.401548 |

ABDE4, pTC13 and PA8 singleton-mean changes from P536:

- alpha 0.25: ABDE4 +2.680642, pTC13 +0.954315, PA8 +0.711165
- alpha 0.50: ABDE4 +1.664110, pTC13 +0.777760, PA8 +0.494986
- alpha 0.75: ABDE4 +1.051780, pTC13 +0.601661, PA8 +0.278487

## D. Clean28 reactions

Alpha 0.25 versus IID t70: 7 improved, 21 worsened, 0 unchanged. Versus J251: 21 improved, 7 worsened, 0 unchanged. Clean28 minus IID +0.177488; minus J251 -0.539612; minus historical best +0.187454.
Improved versus IID t70: ACONF-10, BSR36-31, DC13-1, DIPCS10-7, HAL59-40, S66-50, W4-11-30.
Largest contribution increases versus IID t70: Amino20x4-28 +0.799, PX13-9 +0.561, BHROT27-16 +0.516, BUT14DIOL-13 +0.481, BHPERI-11 +0.338.
Largest contribution decreases versus J251: Amino20x4-28 -2.339, PX13-9 -1.706, BHROT27-16 -1.488, BUT14DIOL-13 -1.384, Amino20x4-54 -1.338.

Alpha 0.50 versus IID t70: 7 improved, 21 worsened, 0 unchanged. Versus J251: 21 improved, 7 worsened, 0 unchanged. Clean28 minus IID +0.359081; minus J251 -0.358020; minus historical best +0.369047.
Improved versus IID t70: ACONF-10, BSR36-31, DC13-1, DIPCS10-7, HAL59-40, S66-50, W4-11-30.
Largest contribution increases versus IID t70: Amino20x4-28 +1.589, PX13-9 +1.126, BHROT27-16 +1.022, BUT14DIOL-13 +0.952, Amino20x4-54 +0.780.
Largest contribution decreases versus J251: Amino20x4-28 -1.549, PX13-9 -1.141, BHROT27-16 -0.982, BUT14DIOL-13 -0.913, Amino20x4-54 -0.876.

Alpha 0.75 versus IID t70: 7 improved, 21 worsened, 0 unchanged. Versus J251: 21 improved, 7 worsened, 0 unchanged. Clean28 minus IID +0.538991; minus J251 -0.178110; minus historical best +0.548957.
Improved versus IID t70: ACONF-10, BSR36-31, DC13-1, DIPCS10-7, HAL59-40, S66-50, W4-11-30.
Largest contribution increases versus IID t70: Amino20x4-28 +2.369, PX13-9 +1.695, BHROT27-16 +1.518, BUT14DIOL-13 +1.414, Amino20x4-54 +1.226.
Largest contribution decreases versus J251: Amino20x4-28 -0.769, PX13-9 -0.572, BHROT27-16 -0.486, BUT14DIOL-13 -0.451, Amino20x4-54 -0.430.

## E. Response across the frozen grid

The five sampled points, including the two reused endpoints, are clean28 monotonic, relchem monotonic, ae17 monotonic, exc nonmonotonic, op nonmonotonic. This grid has a usable eligible candidate. Three alphas are not a continuous Pareto frontier, and Clean28 does not measure an untouched future test.

## F. Next experiment

Next experiment, not run: one read-only reload of the saved alpha 0.75 state `c4809c43eecea209bc0d6e4fd69d76df9679f3de4ec3cd2e2a32efaf36347cea` under the same manifest, Clean28 split, and full90 populations, only if that state is going to be used further. Do not search another alpha. Clean28 and relchem are monotonic on this chord, and alpha 0.50 remains ineligible because its relchem ratio is above 1. Do not fine-tune. Expected cost is about 7 minutes on one local GPU.

## Source hashes

- `train_lap_microbatch.py` `2e837c3a88ca3d4737c3c3397dcfbb98adeb7f6917adc8c693cecf205dc37e9e`
- `train_models/NN_models_lap.py` `862f1a0989b188a9543a49947550521541c7012179ebda0929c92ed824c09cb0`
- `train_models/lap_checkpoint.py` `5ce857c565d94f7538a56104f68894ee59147e0caa3f9f4f098f7ec3abdc418d`
- `train_models/lap_vxc.py` `ca22ac54ae9ffe4d2593f1e7072b11210277b0a321701d5240f564803711c287`
- `train_models/lap_training.py` `ca789703e536f9311178ada07fd41d92ba7f79afd35b25ad240647747b3e9b77`
- `train_models/lap_moo_training.py` `e3dafbe76279e13b73f2a6dd5f2547294902387d21fc1b78b4ce917430495113`
- `train_models/lap_operator.py` `7fae09c0857187875377fbf5c1c7c8d19b1afc869942310f75f5462f7bd2e864`
- `train_models/lap_fixed_adamw.py` `9effc074bd18236ad7af4c4e05fa04be1754d35e228b9a0567163c4b0bd5eb0a`
- `tools/evaluate_microbatch_endpoint.py` `b92a00bc8f51149e29be846bbadbedc74c47fd172b975f1fd0888fd4caf6da3f`
