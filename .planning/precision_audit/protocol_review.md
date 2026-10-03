# Precision protocol review

**Finding:** The original category-B comparison could confound objective precision with step-size changes: the scratch path independently solves FS, while cosine alignment does not constrain direction norm.

**Required control:** At the existing points `t=2^-15` and `t=2^-20`, evaluate scratch-float64 losses at the exact float32 realized trial parameters widened to float64, using the same widened float32-loaded inputs and scratch base. Keep the independent scratch-FS five-point series as a separate diagnostic, and report its direction-norm ratio. Use the matched-state responses for category-B attribution.

**Category-C boundary:** Float32/float64 gradient or direction disagreement indicates arithmetic sensitivity; alone it is not a gradient/objective defect or category C. A finite one-sided residual can reflect curvature and cannot establish C. Support C only with an exact code-path mismatch or independently verified derivative inconsistency after precision and truncation effects are excluded. The five one-sided points may leave C unresolved; use E in that case.

**Disposition:** Root accepted the matched-displacement gap and instructed the executor to add these controls. No source or experiment changes made by this review.
