---
title: "cloelike: A Python Likelihood Library for Cosmological Inference with Euclid Data"
tags:
  - Python
  - cosmology
  - Euclid
  - likelihood
  - weak lensing
  - galaxy clustering
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
    orcid: 0000-0002-4751-5138
    affiliation: 4

  - name: Chiara Moretti
    orcid: 0000-0003-3314-8936
    affiliation: 5

  - name: Andrea Pezzotta
    orcid: 0000-0003-0726-2268
    affiliation: 6

  - name: Zahra Baghkhani
    orcid: 0000-0002-6632-2614
    affiliation: 7

  - name: Klara Bertmann
    orcid: 0009-0004-6700-2470
    affiliation: 8

  - name: Ben Bose
    orcid: 0000-0003-1965-8614
    affiliation: [9, 10]

  - name: Jip de Buck
    orcid: 0000-0000-0000-0000
    affiliation: 2

  - name: Lisa Goh
    orcid: 0000-0002-0104-8132
    affiliation: [9, 10]

  - name: Ivan Sladoljev
    orcid: 0009-0002-9702-2101
    affiliation: 11

  - name: Davide Sciotti
    orcid: 0009-0008-4519-2620
    affiliation: 12

  - name: Nicolas Tessore
    orcid: 0000-0002-9696-7931
    affiliation: 13

  - name: Peter L. Taylor
    orcid: 0000-0001-6999-4718
    affiliation: 14

  - name: Isaac Tutusaus
    orcid: 0000-0002-3199-0399
    affiliation: [7, 15, 13]

affiliations:
  - name: Waterloo Centre for Astrophysics, University of Waterloo, Waterloo, ON N2L 3G1, Canada
    index: 1
  - name: Leiden Observatory, Leiden University, PO Box 9513, 2300 RA, Leiden, the Netherlands
    index: 2
  - name: Centre for Astrophysics Research, University of Hertfordshire, United Kingdom
    index: 3
  - name: German Aerospace Center (DLR), Scientific Information, Linder Höhe, D-51147 Köln, Germany
    index: 4
  - name: INAF - Osservatorio Astronomico di Trieste, Italy
    index: 5
  - name: INAF - Osservatorio Astronomico di Brera, Italy
    index: 6
  - name: Institute of Space Sciences (ICE, CSIC), Campus UAB, Carrer de Can Magrans, s/n, 08193 Barcelona, Spain
    index: 7
  - name: TBD
    index: 8
  - name: INAF - Institute of Space Astrophysics and Cosmic Physics (IASF Milano), Via Corti 12, I-20133 Milano (MI), Italy
    index: 9
  - name: TBD
    index: 10
  - name: Department of Physics, Royal Holloway, University of London, Egham Hill, Egham, UK
    index: 11
  - name: TBD
    index: 12
  - name: Center for Cosmology and AstroParticle Physics (CCAPP), The Ohio State University, Columbus, OH 43210, USA
    index: 13
  - name: Center for Astrophysics and Cosmology, University of Nova Gorica, 1280 Nova Gorica, Slovenia
    index: 14
  - name: Department of Physics & Astronomy, University of Sussex, Brighton BN1 9QH, UK
    index: 15

date: 26 April 2026
bibliography: paper.bib
---

# Summary

\texttt{cloelike}, available at \href{https://github.com/cloe-org/cloelike}{cloe-org/cloelike}, is a Python package providing modular, composable Gaussian likelihood classes for the main cosmological observables targeted by the \emph{Euclid} space mission [@Euclid:2024]. It is a core component of the \textbf{CLOE} (Cosmology Likelihood for Observables in Euclid) ecosystem and interfaces directly with \texttt{cloelib} [@cloelib] for theoretical predictions and \texttt{euclidlib} [@euclidlib] for reading official \emph{Euclid} data products. The package implements Gaussian likelihoods covering angular power spectra ($C_\ell$s) and real-space two-point correlation functions for Weak Lensing (WL), Photometric Galaxy Clustering (GCph), and Galaxy–Galaxy Lensing (GGL) in all joint probe combinations (3×2pt, 2×2pt), as well as spectroscopic power spectrum multipoles with analytical marginalisation over linear nuisance parameters, Baryon Acoustic Oscillations (BAO), and $C_\ell$ likelihoods with a B-mode Nulling Transform (BNT) applied to suppress intrinsic-alignment contamination [@Joachimi:2009; @Taylor:2021]. Probe-specific logic is encapsulated in independent mixins (e.g., `WLMixin`, `GCphMixin`, `GGLMixin`) composed with a shared `PhotoLikelihoodBase`, making it straightforward to add new probes or modify existing ones without code duplication. \texttt{cloelike} is actively used in internal \emph{Euclid} Consortium analyses and is openly released to support community validation and reproducibility.

# Statement of Need

The ESA \emph{Euclid} mission is producing high-precision weak-lensing and galaxy-clustering data that require robust, validated, and reproducible likelihood implementations for cosmological parameter inference [@Euclid:2024]. Despite the broad landscape of cosmological inference tools [@2015-Cosmosis; @2021-Cobaya; @2019-Chisari-CCL], existing public likelihood codes are often probe-specific, or lack the modularity and native data-format support needed for seamless end-to-end \emph{Euclid} analyses.

\texttt{cloelike} addresses this by providing a unified Python interface that:

1. Covers all \emph{Euclid} primary photometric and spectroscopic probes under a single, consistent API.
2. Decouples theoretical predictions (via \texttt{cloelib}) from likelihood evaluation, enabling the backend cosmological model to be swapped without modifying the likelihood code.
3. Reads \emph{Euclid}-format data products natively through \texttt{euclidlib}, eliminating bespoke data-parsing boilerplate.
4. Is suitable for both MCMC samplers (e.g., \texttt{Cobaya} [@2021-Cobaya], \texttt{CosmoSIS} [@2015-Cosmosis], \texttt{nautilus} [@nautilus]) and Fisher-matrix forecasting.

By separating the concerns of data ingestion, scale-cut masking, covariance handling, and observable computation, \texttt{cloelike} lowers the barrier for community members to reproduce official \emph{Euclid} results and to explore new systematic models or probe combinations. The package is used in internal \emph{Euclid} Consortium analyses and constitutes the reference likelihood module for the CLOE analysis pipeline.

# State of the Field

The landscape of cosmological likelihood and inference frameworks includes \texttt{CosmoSIS} [@2015-Cosmosis], \texttt{Cobaya} [@2021-Cobaya], \texttt{CosmoLike} [@CosmoLike], \texttt{CCL} [@2019-Chisari-CCL], and survey-specific implementations such as those used for \emph{KiDS} [@2025-KiDS_Legacy], DES [@2026-DES], and HSC [@2023-HSC]. These tools provide flexible pipelines for parameter estimation and model comparison, but they typically require custom wrappers or interfaces to handle \emph{Euclid}-specific data formats and probe combinations.

\texttt{cloelike} complements these frameworks by offering native support for \emph{Euclid} data products and likelihoods across all primary probes. It can be integrated as a drop-in likelihood module within \texttt{CosmoSIS} or \texttt{Cobaya} workflows, enabling users to leverage advanced sampling capabilities while benefiting from the validated, probe-complete likelihoods that \texttt{cloelike} provides. To the best of our knowledge, \texttt{cloelike} is the first publicly available Python likelihood library to offer coverage of all major \emph{Euclid} photometric and spectroscopic probes under a single composable mixin architecture with native \texttt{euclidlib} data integration.

# Software Design

The architecture of \texttt{cloelike} is built around a clear separation of concerns: data ingestion and covariance handling are managed by a common base class, while probe-specific observable computation is encapsulated in independent mixins. This design mirrors the protocol-based modularity of \texttt{cloelib} [@cloelib], to which \texttt{cloelike} directly interfaces.

## Base Class and Protocol

The `PhotoLikelihoodBase` class provides all functionality shared across photometric likelihoods: data ingestion from \texttt{euclidlib}-format dictionaries, covariance matrix loading and inversion (with `lru_cache` to avoid redundant computation), scale-cut masking via per-bin $\ell$- or $k$-interval filters, geometric rebinning of $C_\ell$ data vectors, and support for coupled (pseudo-$C_\ell$ with mixing matrix) and uncoupled modes. A `PhotoLikelihoodProtocol` (PEP 544 `@runtime_checkable`) defines the required interface, ensuring type safety and enabling compliance checks at runtime.

The shared API for all photometric likelihoods is:

```python
like.get_data_vector_full()        # full data vector
like.get_data_vector_masked()      # masked data vector (cached)
like.get_theory_vector_full(params)  # full theory prediction
like.get_theory_vector_masked(params)  # masked theory prediction (cached)
like.get_covariance_matrix_full()  # full covariance matrix
like.get_covariance_matrix_masked_inv()  # inverse of masked covariance (cached)
like.loglike(params)               # Gaussian log-likelihood
```

## Probe Mixins

Probe-specific logic is encapsulated in mixin classes that can be freely composed:

- **`WLMixin`**: Weak Lensing (SHE×SHE $C_\ell$s or $\xi_+$/$\xi_-$). Handles intrinsic alignment (NLA model: $A_\mathrm{IA}$, $\eta_\mathrm{IA}$, $C_\mathrm{IA}$), multiplicative shear bias, and photometric redshift shift/stretch nuisance parameters.
- **`GCphMixin`**: Photometric Galaxy Clustering (POS×POS $C_\ell$s or $w(\theta)$). Handles polynomial galaxy bias, magnification bias, and photometric redshift nuisance parameters.
- **`GGLMixin`**: Galaxy–Galaxy Lensing (POS×SHE $C_\ell$s or $\gamma_t$). Combines nuisance parameters from both shear and position tracers.
- **`BNTMixin`**: Applies a precomputed B-mode Nulling Transform matrix to shear-related data and theory blocks, suppressing sensitivity to intrinsic alignments at the cost of mixing redshift bins [@Taylor:2021].

Concrete likelihood classes are formed by composing these mixins with `PhotoLikelihoodBase` via Python's multiple inheritance and MRO:

```python
class EuclidLikelihood_3x2pt(WLMixin, GCphMixin, GGLMixin, PhotoLikelihoodBase):
    pass

class EuclidLikelihood_3x2pt_BNT(BNTMixin, WLMixin, GCphMixin, GGLMixin, PhotoLikelihoodBase):
    pass
```

This pattern yields all supported combinations—`EuclidLikelihood_WL`, `EuclidLikelihood_GCph`, `EuclidLikelihood_GGL`, `EuclidLikelihood_2x2pt`, `EuclidLikelihood_3x2pt`, and BNT variants of each—without code duplication.

## Spectroscopic Likelihoods

Spectroscopic likelihoods follow a parallel but independent design:

- **`EuclidLikelihood_GCspectro_Pls`**: Gaussian likelihood for power spectrum multipoles ($\ell = 0, 2, 4$) computed via \texttt{cloelib}'s `LegendreMultipoles` module. Supports Alcock–Paczyński corrections, shot-noise parameters, spectroscopic redshift errors, and survey purity. Optional analytical marginalisation over linear bias parameters (e.g., $b_{\Gamma_3}$, $c_0$, $c_2$, $c_4$, $c_\mathrm{nlo}$, shot-noise terms) using Gaussian priors, following the approach of [@2020JCAP...09..052B].
- **`EuclidLikelihood_BAO`**: Gaussian likelihood for BAO $\alpha_\parallel$ and $\alpha_\perp$ parameters, computed via \texttt{cloelib}'s `BaryonAcousticOscillations` module against a fiducial cosmology.

Both spectroscopic classes accept protocol-compliant `Background` and `SpectroPower` (or `Background`-only for BAO) objects from \texttt{cloelib}, enabling the cosmological backend to be swapped transparently.

## Redshift Distribution Nuisance Parameters

Photometric redshift calibration is modelled via per-bin shift ($\Delta z_i$) and stretch ($w_i$) parameters applied to the galaxy redshift distributions $n(z)$, following the standard approach used in current Stage-III surveys [@2025-KiDS_Legacy; @2026-DES].

# Usage Examples

## Photometric $C_\ell$ Likelihood (3×2pt)

```python
from cloelike import EuclidLikelihood_3x2pt
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuNonLinearPerturbations

# Instantiate the likelihood with data and settings
like = EuclidLikelihood_3x2pt(
    data=data,           # euclidlib-format data dict (cells, cov, dndz, z_arr, ells, mixmat)
    settings=settings,   # scale_cuts, n_ell_bins, etc.
    Background=CAMBBackground,
    LinPerturbations=CAMBLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
    mode="coupled",      # use pseudo-Cl with mixing matrix
)

# Compute log-likelihood for a given parameter set
loglike = like.loglike(parameters)

# Inspect the masked data and theory vectors
data_vec  = like.get_data_vector_masked()
th_vec    = like.get_theory_vector_masked(parameters)
```

Switching to a BNT analysis requires only a class change:

```python
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt_BNT

like_bnt = EuclidLikelihood_3x2pt_BNT(
    data=data,    # data["BNT_matrix"] must be provided
    settings=settings,
    Background=CAMBBackground,
    LinPerturbations=CAMBLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
)
loglike_bnt = like_bnt.loglike(parameters)
```

## Two-Point Correlation Function Likelihood

The real-space 2PCF likelihoods in `EuclidLikelihood_photo_2pcf` follow the same mixin structure. For WL, the likelihood covers both $\xi_+$ and $\xi_-$:

```python
from cloelike.EuclidLikelihood_photo_2pcf import EuclidLikelihood_WL as EuclidLikelihood_WL_2pcf

like_2pcf = EuclidLikelihood_WL_2pcf(
    data=data,      # data["2pcf"], data["theta"] (arcmin), data["cov"], data["dndz_she"]
    settings=settings,  # scale_cuts per bin pair: [theta_min_plus, theta_max_plus, theta_min_minus, theta_max_minus]
    Background=CAMBBackground,
    LinPerturbations=CAMBLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
    ells_integration=np.arange(2, 40000),
)
loglike_2pcf = like_2pcf.loglike(parameters)
```

## Spectroscopic Power Spectrum Multipoles

```python
from cloelike import EuclidLikelihood_GCspectro_Pls
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower

like_spectro = EuclidLikelihood_GCspectro_Pls(
    data=data,          # data["GCspectro"][z_eff]: pk0, pk2, pk4, cov, k, nbar, ...
    settings=settings,  # scale_cuts per bin and multipole
    Background=CAMBBackground,
    SpectroPower=CometEFT_SpectroPower,
    AM_priors={         # optional: analytical marginalisation priors
        1.0: {"c0": [0.0, 10.0], "c2": [0.0, 10.0]},
    },
)
loglike_spectro = like_spectro.loglike(parameters)
```

## BAO Likelihood

```python
from cloelike import EuclidLikelihood_BAO
from cloelib.cosmology.camb_cosmology import CAMBBackground

like_bao = EuclidLikelihood_BAO(
    data=data,       # data["BAO"][z_eff]: alpha_parallel, alpha_perp, covariance, fiducial_cosmology
    Background=CAMBBackground,
)
loglike_bao = like_bao.loglike(parameters)
```

## Joint Likelihood

Probe likelihoods can be combined to form a joint log-likelihood:

```python
total_loglike = like.loglike(parameters) + like_spectro.loglike(parameters) + like_bao.loglike(parameters)
```

This can be directly passed to any sampler (e.g., \texttt{nautilus} [@nautilus], \texttt{Cobaya} [@2021-Cobaya]) as a callable.

# Documentation

Comprehensive documentation for \texttt{cloelike} is available at \href{https://cloe-org.github.io/cloelike/dev/}{cloe-org.github.io/cloelike/dev/}. The documentation includes API references, installation instructions, and a description of the software structure.

For practical examples and interactive tutorials, the \href{https://github.com/cloe-org/playground}{cloe-org/playground} repository hosts Jupyter notebooks showcasing typical use cases including single-probe and joint analyses, blinding workflows, and integration with popular MCMC samplers.

\texttt{cloelike} input and output formats for both photometric and spectroscopic observables are fully compatible with \texttt{euclidlib} data products.

# Availability

**Source:** [github.com/cloe-org/cloelike](https://github.com/cloe-org/cloelike)
**License:** MIT
**Install:** \texttt{pip install cloelike}
**Documentation:** [https://cloe-org.github.io/cloelike/dev/](https://cloe-org.github.io/cloelike/dev/)
**Examples:** [github.com/cloe-org/playground](https://github.com/cloe-org/playground)

# Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors (Bonici, Cañas-Herrera, Carrilho, Casas, Moretti, Pezzotta) are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of \texttt{cloelike}, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: Review of software architecture and design.
- **G. Cañas-Herrera**: Repository set-up and overall software architecture; implementation of `PhotoLikelihoodBase` including data ingestion, scale-cut masking, covariance inversion, geometric rebinning of $C_\ell$ data vectors, and `lru_cache` optimisation; mixin composition framework; CI pipeline configuration; pre-commit and code-quality tooling; issue and pull-request templates; README, documentation infrastructure, and community contribution tracking (\texttt{all-contributors}); pyproject.toml versioning and release workflows; BAO unit tests.
- **P. Carrilho**: Implementation of systematic nuisance parameters for photometric tracers (multiplicative shear bias, photometric redshift shifts and width parameters); coupled/uncoupled mode flag for pseudo-$C_\ell$ likelihood with mixing matrices; photometric likelihood refinements and notebook tutorials.
- **S. Casas**: Review of design and ideas.
- **C. Moretti**: Implementation of `EuclidLikelihood_BAO` (BAO $\alpha$-parameter likelihood); spectroscopic likelihood bug fixes; unit tests for BAO likelihood; repository housekeeping.
- **A. Pezzotta**: Core implementation of `EuclidLikelihood_GCspectro_Pls` (full-shape spectroscopic power spectrum multipoles); analytical marginalisation over linear bias parameters; spectroscopic data-vector and covariance-matrix handling; unit tests for spectroscopic likelihood; homogenisation of photometric and spectroscopic likelihood APIs.

The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot. A full, categorised breakdown is available in the \texttt{cloelike} repository README.

# Acknowledgements

We thank the broader CLOE software development team for the foundational work that motivated this library. G.C.H. acknowledges that this project is part of the UNICORN project (file number VI.Veni.242.110) within the Talent Programme Veni Science domain 2024, which is partly financed by the Dutch Research Council (NWO) under grant [https://doi.org/10.61686/ZCPQI32997](https://doi.org/10.61686/ZCPQI32997). M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). C.M. is supported by the Agenzia Spaziale Italiana project "Attività scientifica per la missione Euclid – fase E ACCORDO ATTUATIVO n. 2024-10-HH.0." B.B. is supported by a UK Research and Innovation Stephen Hawking Fellowship (EP/W005654/2).

We acknowledge the EuroHPC Joint Undertaking for awarding project ID EHPC-EXT-2024E02-083 access to Leonardo, hosted by CINECA (Italy). We acknowledge the use of the Spanish Supercomputing Network (RES) resources provided by the Barcelona Supercomputing Center (BSC) on MareNostrum 5 under allocations AECT-2024-3-0020, 2025-1-0045, 2025-2-0046, and 2025-3-0036.

The Euclid Consortium acknowledges the European Space Agency and a number of agencies and institutes that have supported its development, in particular: the Agenzia Spaziale Italiana; the Austrian Forschungsförderungsgesellschaft funded through BMIMI; the Belgian Science Policy; the Canadian Euclid Consortium; the Deutsches Zentrum für Luft- und Raumfahrt; DTU Space and the Niels Bohr Institute (Denmark); the French Centre National d'Études Spatiales; the Fundação para a Ciência e a Tecnologia; the Hungarian Academy of Sciences; the Ministerio de Ciencia, Innovación y Universidades; the National Aeronautics and Space Administration; the National Astronomical Observatory of Japan; the Netherlands Research School for Astronomy; the Norwegian Space Agency; the Research Council of Finland; the Romanian Space Agency; the Swiss Space Office (SSO) at the State Secretariat for Education, Research, and Innovation (SERI); and the United Kingdom Space Agency. A complete and detailed list is available at [www.euclid-ec.org/consortium/community/](https://www.euclid-ec.org/consortium/community/).

# References
