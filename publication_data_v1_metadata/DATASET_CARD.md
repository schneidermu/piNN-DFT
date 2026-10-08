# Laplacian functional publication dataset v1

268 Minnesota chemical identities (251 relative chemistry, 17 AE17), each with eight grid augmentations. Grid variants never multiply reaction weights. 90 mRKS systems retain all E_xc, gauge-fixed pointwise and weak-form AO targets.

Diet30 fixed PBE0 densities: diagnostic view is not selectable; clean validation excludes reserved/test and potential training overlaps. PBE0-D3(BJ) is primary; PBE-D3(BJ) is secondary. No test split or reserved test labels/arrays are packaged.

Scientific units/dtypes are per-dataset attributes and provenance/schema.json. Chemistry preserves legacy F32 rounding before matched-F64 arithmetic. Operator precision remains the repaired stored-F32/learned-F64/PBE-F32/AO-F64 contract. Validation stores native F64 densities and precomputed grid descriptors. Its non-XC term is kinetic+nuclear attraction+Coulomb+nuclear repulsion, without any exchange/correlation.

The training non-XC scalar is preserved from legacy ener[0]; raw Minnesota checkpoint/grid construction is not reconstructed by this conversion. mRKS Exc is the exact qualified legacy target, not substituted by another source field. Sources are hash-bound. Minnesota corpus originates from the project/University of Minnesota database; Diet definitions originate from gambort/DietGMTKN55. Source redistribution licenses/permissions must be confirmed before external archival; this local bundle does not invent a data license.

Validation qualification: 84/84 species; 82 direct fixed-density total-energy parity + 2 MCONF independent component-level parity. Direct maximum discrepancy 5.684341886080801e-13 Ha; component maximum 1.1368683772161603e-12 Ha. Both MCONF density/descriptor/PBE XC comparisons are exact.

Unique validation exclusions: BH76-5 and G21EA-25; 30 - 2 = 28 selectable reactions. BH76-5 also has a conservative ambiguous Minnesota collision and is counted once.

Frozen and immutable by convention after final atomic publication. Content changes require a new version/hash. No external baseline evaluation was part of dataset construction.
