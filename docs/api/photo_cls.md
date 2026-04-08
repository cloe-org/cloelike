# Photometric Likelihoods

These classes implement Gaussian likelihoods for Euclid photometric primary probes, covering both angular power spectra (Cls) and real-space two-point correlation functions (2PCF).

---

## Base

Shared by both Cls and 2PCF likelihoods.

### PhotoLikelihoodProtocol

::: cloelike.EuclidLikelihood_photo_base.PhotoLikelihoodProtocol
options:
show_source: true

### PhotoLikelihoodBase

::: cloelike.EuclidLikelihood_photo_base.PhotoLikelihoodBase
options:
show_source: true

---

## Angular Power Spectra (Cls)

Classes from `cloelike.EuclidLikelihood_photo_Cls`.

**Mixins** — composable building blocks for Cls likelihoods:

### WLMixin

::: cloelike.EuclidLikelihood_photo_Cls.WLMixin
options:
show_source: true

### GCphMixin

::: cloelike.EuclidLikelihood_photo_Cls.GCphMixin
options:
show_source: true

### GGLMixin

::: cloelike.EuclidLikelihood_photo_Cls.GGLMixin
options:
show_source: true

### BNTMixin

::: cloelike.EuclidLikelihood_photo_Cls.BNTMixin
options:
show_source: true

**Concrete likelihoods:**

### EuclidLikelihood_WL

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_WL
options:
show_source: true

### EuclidLikelihood_GCph

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_GCph
options:
show_source: true

### EuclidLikelihood_GGL

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_GGL
options:
show_source: true

### EuclidLikelihood_3x2pt

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_3x2pt
options:
show_source: true

### EuclidLikelihood_2x2pt

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_2x2pt
options:
show_source: true

**BNT variants** — Cls likelihoods with B-mode Nulling Transform applied to shear blocks. Require a precomputed `data['BNT_matrix']`:

### EuclidLikelihood_WL_BNT

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_WL_BNT
options:
show_source: true

### EuclidLikelihood_GGL_BNT

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_GGL_BNT
options:
show_source: true

### EuclidLikelihood_3x2pt_BNT

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_3x2pt_BNT
options:
show_source: true

### EuclidLikelihood_2x2pt_BNT

::: cloelike.EuclidLikelihood_photo_Cls.EuclidLikelihood_2x2pt_BNT
options:
show_source: true

---

## Two-Point Correlation Functions (2PCF)

Classes from `cloelike.EuclidLikelihood_photo_2pcf`, computing likelihoods in real space (ξ₊, ξ₋, w, γₜ).

**Mixins** — composable building blocks for 2PCF likelihoods:

### WLMixin (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.WLMixin
options:
show_source: true

### GCphMixin (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.GCphMixin
options:
show_source: true

### GGLMixin (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.GGLMixin
options:
show_source: true

**Concrete likelihoods:**

### EuclidLikelihood_WL (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.EuclidLikelihood_WL
options:
show_source: true

### EuclidLikelihood_GCph (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.EuclidLikelihood_GCph
options:
show_source: true

### EuclidLikelihood_GGL (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.EuclidLikelihood_GGL
options:
show_source: true

### EuclidLikelihood_3x2pt (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.EuclidLikelihood_3x2pt
options:
show_source: true

### EuclidLikelihood_2x2pt (2PCF)

::: cloelike.EuclidLikelihood_photo_2pcf.EuclidLikelihood_2x2pt
options:
show_source: true
