---
title: "cloelike: A Python Likelihood Library for Computing Cosmological Observables in the Euclid Era"
tags:
  - Python
  - cosmology
  - Euclid
  - Likelihood
authors:
  - name: Marco Bonici
    orcid: 0000-0002-8430-126X
    affiliation: 1
  - name: Guadalupe Cañas-Herrera
    orcid: 0000-0003-2796-2149
    affiliation: 2
  - name: Pedro Carrilho
    orcid: 0000-0003-1339-0194
    affiliation: 3
  - name: Santiago Casas
    orcid: 0000-0000-0000-0000
    affiliation: 4
  - name: Chiara Moretti
    orcid: 0000-0003-3314-8936
    affiliation: 5
  - name: Andrea Pezzotta
    orcid: 0000-0003-0726-2268
    affiliation: 6
affiliations:
  - name: University of Waterloo, Canada
    index: 1
  - name: Leiden Observatory, the Netherlands
    index: 2
  - name: Centre for Astrophysics Research, University of Hertfordshire, United Kingdom
    index: 3
  - name: RWTH Aachen University, Germany
    index: 4
  - name: INAF - Osservatorio Astronomico di Trieste, Italy
    index: 5
  - name: INAF - Osservatorio Astronomico di Brera, Italy
    index: 6
date: 8 April 2026
bibliography: paper.bib
---

# Summary

`cloelike` is a Python package providing modular, composable Gaussian likelihood classes for the primary photometric and spectroscopic observables of the _Euclid_ space mission [@Euclid:2024]. It is part of the **CLOE** (Cosmology Likelihood for Observables in Euclid) ecosystem, interfacing with `cloelib` for theoretical predictions and `euclidlib` for reading official Euclid data products.

The package implements likelihoods for:

- **Angular power spectra (Cls)**: Weak Lensing (WL), Photometric Galaxy Clustering (GCph), Galaxy–Galaxy Lensing (GGL), and their joint combinations (3×2pt, 2×2pt).
- **Two-point correlation functions (2PCF)**: Real-space counterparts of the above ($\xi_+$, $\xi_-$, $w$, $\gamma_t$).
- **Spectroscopic Galaxy Clustering (GCspectro)**: Power spectrum multipoles and Baryon Acoustic Oscillations (BAO).
- **B-mode Nulling Transform (BNT)**: Cls likelihoods with a BNT applied to shear-related blocks to suppress intrinsic-alignment contamination.

The design follows a mixin-based architecture: probe-specific building blocks (e.g., `WLMixin`, `GCphMixin`, `GGLMixin`) are composed with a shared `PhotoLikelihoodBase` to form concrete likelihood classes, making it straightforward to add new probes or modify existing ones without code duplication.

# Statement of Need

The _Euclid_ satellite [@Euclid:2024] is producing unprecedented weak-lensing and galaxy-clustering surveys that require robust, validated, and reproducible likelihood implementations for cosmological parameter inference. Existing public codes either target specific probes or lack the modular flexibility needed to swap observables and covariance matrix inputs in production analyses.

`cloelike` addresses this gap by providing a unified Python interface that:

1. Covers all Euclid primary photometric and spectroscopic probes under a single, consistent API.
2. Decouples theoretical predictions (delegated to `cloelib`) from the likelihood evaluation logic.
3. Natively reads Euclid-format data products via `euclidlib`, ensuring compatibility with official data releases.
4. Is designed for use within both MCMC samplers and Fisher-matrix forecasting pipelines.

The package has been used in internal Euclid Consortium analyses and is released openly to facilitate community validation and reproducibility of Euclid cosmological results.

# API Design

Likelihood classes share a common base, `PhotoLikelihoodBase`, which handles data ingestion and covariance matrix loading. Probe-specific functionality is encapsulated in mixins:

```python
from cloelike import EuclidLikelihood_3x2pt

like = EuclidLikelihood_3x2pt(data=data, settings=settings)
loglike = like.loglike(theory_vector)
```

BNT variants simply add the `BNTMixin` to the MRO:

```python
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt_BNT

like_bnt = EuclidLikelihood_3x2pt_BNT(data=data, settings=settings)
```

This design ensures that switching between standard and BNT analyses requires changing only the class, not the surrounding inference code.

## Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of `cloelib`, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: TBA
- **G. Cañas-Herrera**: TBA
- **P. Carrilho**: TBA
- **S. Casas**: TBA
- **C. Moretti**: TBA
- **A. Pezzotta**: TBA

The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot, following the specification of the same name. A full, categorised breakdown of each contributor's role—including code, documentation, testing, ideas, project management, and more—is available in the `README` of the `cloelib` repository.

# Acknowledgements

We acknowledge the support of the Euclid Consortium. We thank the broader CLOE software development team for foundational work that motivated this library. GCH acknowledges that this project is part of the project UNICORN with file number VI.Veni.242.110 of the research programme Talent Programme Veni Science domain 2024 which is (partly) financed by the Dutch Research Council (NWO) under the grant https://doi.org/10.61686/ZCPQI32997. M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC Extreme Access call.

# References
