---
title: "cloelike: A Python Library for Cosmological Likelihood Inference in the Euclid Era"
tags:
  - Python
  - cosmology
  - Euclid
  - likelihood
  - weak lensing
  - galaxy clustering
authors:
  # cloe-org maintainers
  - name: Marco Bonici
    orcid: 0000-0002-8430-126X
    affiliation: 1

  - name: Guadalupe Cañas-Herrera
    orcid: 0000-0003-2796-2149
    corresponding: true
    affiliation: 2

  - name: Pedro Carrilho
    orcid: 0000-0003-1339-0194
    affiliation: 3

  - name: Santiago Casas
    orcid: 0000-0002-4751-5138
    affiliation: [4, 17]

  - name: Chiara Moretti
    orcid: 0000-0003-3314-8936
    affiliation: 5

  - name: Andrea Pezzotta
    orcid: 0000-0003-0726-2268
    affiliation: 6

  # cloe-org contributors
  - name: Zahra Baghkhani
    orcid: 0000-0002-6632-2614
    affiliation: 7

  - name: Klara Bertmann
    orcid: 0009-0004-6700-2470
    affiliation: 9

  - name: Carmelita Carbone
    orcid: 0000-0003-0125-3563
    affiliation: 8

  - name: Martin Crocce
    orcid: 0000-0002-9745-6228
    affiliation: 7

  - name: Jip de Buck
    orcid: 0009-0001-5175-9282
    affiliation: 2

  - name: Nastassia Grimm
    orcid: 0000-0001-9602-0599
    affiliation: 10

  - name: Martin Kärcher
    orcid: 0000-0001-5868-647X
    affiliation: 11

  - name: Felicitas Keil
    orcid: 0000-0002-8108-1679
    affiliation: 12

  # TODO: add ORCID
  - name: Zuzanna Kostrzewa-Rutkowska
    affiliation: 2

  - name: Raphaël Kou
    orcid: 0000-0003-3408-3062
    affiliation: 12

  - name: Davide Sciotti
    orcid: 0009-0008-4519-2620
    affiliation: 13

  - name: Ivan Sladoljev
    orcid: 0009-0002-9702-2101
    affiliation: 18

  - name: Peter L. Taylor
    orcid: 0000-0001-6999-4718
    affiliation: 14

  - name: Nicolas Tessore
    orcid: 0000-0002-9696-7931
    affiliation: 15

  - name: Isaac Tutusaus
    orcid: 0000-0002-3199-0399
    affiliation: [7, 16, 12]

  - name: Casper Vedder
    orcid: 0009-0007-6341-4648
    affiliation: 2

affiliations:
  - name: Waterloo Centre for Astrophysics, University of Waterloo, Canada
    index: 1
  - name: Leiden Observatory, Leiden University, Netherlands
    index: 2
  - name: Centre for Astrophysics Research, University of Hertfordshire, United Kingdom
    index: 3
  - name: Scientific Information, German Aerospace Center (DLR), Germany
    index: 4
  - name: INAF - Osservatorio Astronomico di Trieste, Italy
    index: 5
  - name: INAF - Osservatorio Astronomico di Brera, Italy
    index: 6
  - name: Institute of Space Sciences (ICE, CSIC), Spain
    index: 7
  - name: INAF - IASF Milano, Italy
    index: 8
  - name: Astronomical Institute (AIRUB), Ruhr University Bochum, Germany
    index: 9
  - name: Department of Physics, University of Oxford, United Kingdom
    index: 10
  - name: Dipartimento di Fisica "Aldo Pontremoli", Università degli Studi di Milano, Italy
    index: 11
  - name: IRAP, Université de Toulouse, France
    index: 12
  - name: Osservatorio Astronomico di Roma, Italy
    index: 13
  - name: CCAPP, The Ohio State University, USA
    index: 14
  - name: Mullard Space Science Laboratory, University College London, United Kingdom
    index: 15
  - name: Institut d'Estudis Espacials de Catalunya (IEEC), Spain
    index: 16
  - name: TTK, RWTH Aachen University, Germany
    index: 17
  - name: Department of Physics, Royal Holloway, University of London, United Kingdom
    index: 18

date: 16 October 2026
bibliography: paper.bib
---

# Summary

\texttt{cloelike}, available at \href{https://github.com/cloe-org/cloelike}{cloe-org/cloelike}, is a Python package providing modular, composable Gaussian likelihood classes for the main cosmological large-scale structure observables targeted by the ESA \emph{Euclid} space mission. It is a core component of the \texttt{CLOE} (Cosmology Likelihood for Observables in Euclid) ecosystem and interfaces directly with \texttt{cloelib} for theoretical predictions and \texttt{euclidlib} for reading official \emph{Euclid} data products. Here we present version 0.3 of \texttt{cloelike}, the first release distributed through PyPI. It implements Gaussian likelihoods covering harmonic angular power spectra, real-space two-point correlation functions, and COSEBIs for weak lensing (WL), photometric galaxy clustering (GCph), and galaxy--galaxy lensing (GGL) in all joint probe combinations ($3\times2$pt, $2\times2$pt), as well as spectroscopic galaxy clustering through full-shape power spectrum and correlation function multipoles, baryon acoustic oscillations (BAO), and their joint full-shape+BAO combination. \texttt{cloelike} is actively used in internal Euclid Consortium analyses and is openly released to support community validation and reproducibility.

# Statement of Need

The ESA \emph{Euclid} mission is delivering high-precision photometric and spectroscopic observations aimed at probing the nature of dark energy and dark matter, as well as constraining the origin of primordial density perturbations in the early Universe. Achieving these scientific goals requires confronting (extended) cosmological models with the data through robust, validated, and reproducible likelihood frameworks for statistical inference [@Euclid:2024; @EP-CLOE3]. Despite the wide ecosystem of cosmological inference tools, such as \texttt{CosmoSIS} [@2015-Cosmosis] and \texttt{Cobaya} [@2021-Cobaya], most publicly available likelihood implementations remain probe-specific or lack the modularity and native data-format compatibility needed for fully integrated end-to-end \emph{Euclid} analyses.

\texttt{cloelike} addresses this limitation by providing a minimal and unified Python interface dedicated exclusively to computing a $\chi^2$ statistic. It operates on a data vector, covariance matrix supplied externally, and theoretical predictions supplied by \texttt{cloelib}, all consistently structured within the \texttt{euclidlib}[^euclidlib] data format. This strict separation of concerns, combined with seamless integration, ensures interoperability while avoiding duplication of functionality already provided by external inference frameworks or posterior samplers.

[^euclidlib]: <https://github.com/euclidlib/euclidlib>

Concretely, \texttt{cloelike} evaluates the standard multivariate Gaussian likelihood through the quadratic form
$$-2\ln \mathcal{L} = (\mathbf{d} - \mathbf{t})^{\mathrm{T}} \mathbf{C}^{-1} (\mathbf{d} - \mathbf{t}) + b,$$
where $\mathbf{d}$ denotes the data vector, $\mathbf{t}$ the theoretical prediction, $\mathbf{C}$ the covariance matrix, and $b$ is an additive bias. This quantity fully specifies the Gaussian log-likelihood up to an additive normalization constant and constitutes the essential input required by most inference engines.

By design, \texttt{cloelike} is a lightweight and modular package that focuses exclusively on likelihood evaluation. It can be seamlessly interfaced with established Bayesian frameworks such as \texttt{Cobaya} or \texttt{CosmoSIS}, as well as integrated into custom sampling pipelines that operate on $\chi^2$ evaluations. Core tasks such as parameter exploration, sampling strategies, and model management are deliberately delegated to external tools, ensuring that \texttt{cloelike} remains dedicated to providing a consistent, validated, and reproducible likelihood computation.

This modular architecture enforces a clear separation between data handling, theoretical prediction, and statistical inference. In turn, \texttt{cloelike} lowers the barrier for the community to reproduce official \emph{Euclid} results, while enabling flexible exploration of new systematics, extended cosmological models, and probe combinations. The package is actively used within the Euclid Consortium, where it serves as the reference likelihood implementation and represents a natural evolution of the former \texttt{CLOE} software [@EP-CLOE2; @EP-CLOE1; @EP-CLOE4].

# State of the Field

The ecosystem of cosmological likelihood and inference frameworks is both extensive and heterogeneous. Widely used examples include \texttt{CosmoSIS}, which has been broadly adopted across the community and in major galaxy surveys such as the \emph{Dark Energy Survey} (DES); \texttt{Cobaya}, favored among theoretical cosmologists for its flexibility in combining multiple data sets and interfacing with different Boltzmann solvers; \texttt{CosmoLike} [@CosmoLike]; and \texttt{Firecrown}[^firecrown], among others. Furthermore, survey-specific pipelines and wrappers have been developed, such as \texttt{CosmoPipe}[^cosmopipe] for \emph{Kilo Degree Survey} (KiDS) analyses. These frameworks typically provide end-to-end Bayesian analysis environments, combining theoretical predictions, data handling, likelihood evaluation, and interfaces to sampling engines within a unified software structure.

[^firecrown]: <https://github.com/LSSTDESC/firecrown>

[^cosmopipe]: <https://github.com/AngusWright/CosmoPipe>

In contrast, \texttt{cloelike} adopts a deliberately modular and simpler philosophy, closer in spirit to self-contained likelihood implementations such as the \emph{Planck 2018} \texttt{plik} likelihood [@Planck2018]. Rather than embedding the full inference pipeline, \texttt{cloelike} focuses exclusively on providing a validated and consistent likelihood layer with native support for \emph{Euclid} data products across all primary probes. In this way, it complements existing frameworks: it can be seamlessly integrated as a drop-in likelihood module within \texttt{CosmoSIS} or \texttt{Cobaya} workflows, allowing users to exploit advanced sampling capabilities while relying on a robust, probe-complete likelihood implementation. At the same time, its minimal interface based solely on $\chi^2$ evaluation, makes it readily compatible with any external sampler or inference tool that operates on likelihood values, enabling flexible integration within a wide range of analysis pipelines (e.g. \texttt{Nautilus} [@nautilus], \texttt{dynesty} [@Higson2019], \texttt{emcee} [@emcee], and others).

# Software Design

\texttt{cloelike} v0.3 supports likelihoods already prepared for photometric harmonic and configuration-space summary statistics (the latter including two-point correlation functions and COSEBIs), as well as spectroscopic galaxy clustering observables, both in their full-shape (power spectrum and correlation function multipoles) and BAO formulations, with the possibility of constructing a joint likelihood combining the two spectroscopic probes. On the photometric side, v0.3 provides the weak lensing, photometric galaxy clustering, galaxy--galaxy lensing, $3\times2$pt and $2\times2$pt likelihoods, an optional Bernardeau--Nishimichi--Taruya (BNT) transform of the shear-related blocks, a choice of intrinsic alignment models (NLA or TATT) and galaxy bias models (polynomial, per-bin, or non-linear), and nuisance parameters for multiplicative shear bias, magnification bias, and shifts and stretches of the redshift distributions. On the spectroscopic side, full-shape likelihoods can use either the \texttt{COMET} (EFTofLSS or VDG$_\infty$) or \texttt{PBJ} theory models, with analytical marginalisation over the parameters that enter the model linearly, under Gaussian priors. All spectroscopic likelihoods can apply the Hartlap correction to the inverse covariance when it is estimated from a finite number of mocks.

The internal architecture of \texttt{cloelike} follows a modular design based on a strict separation of concerns. For photometric probes, which share a common statistical structure, all operations related to data-vector ingestion, masking, scale cuts, covariance loading, precision-matrix handling, and consistency checks are implemented in a shared base class[^spectrobase]. This common layer provides a unified interface for handling observational inputs independently of the specific summary statistic being analysed. On top of this shared infrastructure, the computation of the theoretical observables is delegated to independent probe-specific mixins. Each mixin encapsulates only the logic required to compute a given observable—for example, harmonic-space angular power spectra, configuration-space correlation functions, or COSEBIs—while inheriting the shared data and covariance machinery from the base class. This approach avoids code duplication, ensures consistency across probes, and makes the addition of new observables straightforward, since only the corresponding theoretical prediction module needs to be implemented.

[^spectrobase]: For spectroscopic galaxy clustering, given its self-standing nature but different from photometric probes, a common class is not necessary.

This architecture closely mirrors the protocol-oriented modularity adopted in \texttt{cloelib}, with which \texttt{cloelike} interfaces directly. In practice, \texttt{cloelike} acts as the likelihood layer responsible for statistical evaluation, while \texttt{cloelib} provides the theoretical predictions through interchangeable protocols. The result is a flexible and extensible framework in which likelihood definitions and theory computations remain cleanly decoupled, facilitating maintenance, validation, and future extensions to new cosmological probes.

Building on this modular foundation, \texttt{cloelike} enables a high degree of flexibility in composing likelihoods across probes and summary statistics. Each likelihood instance is effectively constructed by combining a data container, a covariance representation, and one or more observable providers, allowing users to tailor configurations without modifying the underlying infrastructure. This composability is particularly advantageous in the context of multi-probe survey analyses, where consistency across probes must be maintained while retaining the ability to evolve individual components independently.

A key feature of the framework is its seamless integration with theory predictions provided by \texttt{cloelib} using Python protocols. Through a well-defined interface, \texttt{cloelike} retrieves theoretical predictions for all supported observables, ensuring consistency in cosmological parameter definitions, nuisance modeling, and numerical settings. This tight coupling minimizes duplication of functionality and code, and reduces the risk of inconsistencies between likelihood and theory layers. Moreover, it allows \texttt{cloelike} to benefit directly from any future developments in \texttt{cloelib}. This includes, for example, ongoing improvements in non-linear modeling, such as galaxy bias, redshift-space distortions, and intrinsic alignment, as well as extensions to beyond-$\Lambda$CDM scenarios.

From an implementation perspective, Python mixins in the photometric case provide an elegant yet powerful mechanism to extend the framework for likelihood combinations, such as $3\times2$pt and $2\times2$pt. New probes or summary statistics can be incorporated by defining additional mixins that implement the required observable computations and data-vector mappings, without altering the core likelihood machinery. This promotes rapid prototyping and facilitates the inclusion of novel observables, such as cross-correlations with the cosmic microwave background, which is essential for future data releases and methodological developments within the Euclid Consortium.

All \texttt{cloelike} likelihoods, independently of the probe or data space considered, are initialised from two Python dictionaries that provide the required data products and analysis specifications. The \texttt{data} dictionary contains the data vector, auxiliary data products, and the corresponding covariance matrix, while the \texttt{settings} dictionary defines analysis choices such as scale cuts for each redshift bin in tomographic analyses.

In addition, each likelihood requires the appropriate \texttt{cloelib} protocol-compatible theory classes. Photometric likelihoods require classes for \texttt{Background}, \texttt{LinPerturbations} and \texttt{NonLinPerturbations}; spectroscopic galaxy-clustering full-shape likelihoods require \texttt{Background} and \texttt{SpectroPower} (plus a \texttt{Perturbations} class when the theory model takes the linear power spectrum as input, as \texttt{PBJ} does); and BAO likelihoods require only a \texttt{Background} class. \texttt{cloelike} v0.3 requires \texttt{cloelib} v0.11 or later.

Moreover, \texttt{cloelike} is designed with performance and scalability in mind. Vectorized operations, caching of intermediate quantities (the inverse of the covariance matrix, masking vectors, and the projection kernels and perturbations shared by the probes of a combined likelihood), and compatibility with parallel sampling codes ensure that the likelihood evaluations remain efficient even for high-dimensional parameter spaces. The framework is also compatible with standard sampling and optimization tools, enabling its deployment in both Bayesian inference and frequentist pipelines.

Finally, the architecture naturally supports joint analyses across multiple probes. By construction, shared parameters spaces, as long as there is no cross-covariances between the probes, can be consistently handled, allowing \texttt{cloelike} to perform coherent multi-probe likelihood evaluations by means of summing log-likelihood evaluations. This capability is central to extracting the full scientific potential of forthcoming survey data, where combined constraints from weak lensing, galaxy clustering, and other observables will play a decisive role in testing cosmological models.

# Usage Examples

In this section, we illustrate the initialization and evaluation of a single $\chi^2$ for two representative likelihoods implemented in \texttt{cloelike}. These examples expect to ingest data vectors, covariance matrices, and other data products (i.e: galaxy redshift distributions) in the \texttt{euclidlib} format, as well to specify scale cuts on those data vectors.

## Harmonic photometric likelihoods

In this example, we evaluate the log-likelihood for cosmic shear angular power spectra. For photometric likelihood analyses, \texttt{cloelike} is designed to ingest covariance matrices ordered by "probe", "redshift bin", and "multipole". This convention corresponds to the native output format of the \textit{Euclid} covariance matrix software \texttt{Spaceborne}[^spaceborne].

[^spaceborne]: <https://github.com/davidesciotti/Spaceborne>

```python
# ------------------------------------------------------------------
# Step 1: Construct the data dictionary expected by cloelike
# ------------------------------------------------------------------
# This function assembles the inputs required by the likelihood:
# - Angular power spectra (cls_data)
# - Multipole range (ells)
# - Galaxy Redshift distribution (myz, my_dndz_she_norm)
# - Covariance matrix (cov)
# - Mixing matrix (mixmats), accounting for mode coupling

def build_data_WL(ell_key, cov):
    return {
        'cells': cls_data,
        'ells': cls_data[ell_key].ell,
        'z_arr': myz,
        'cov': cov,
        'mixmat': mixmats,
        'dndz_she': my_dndz_she_norm, # must be normalised
    }


# ------------------------------------------------------------------
# Step 2: Define analysis settings dictionary expected by cloelike
# ------------------------------------------------------------------
# These control how the data vector is compressed and which scales
# are included in the likelihood evaluation.

def build_settings():
    scale_cuts = {key: [5, 3000] for key in cls_data}
    return {
        'n_ell_bins': 32,        # Number of multipole bins
        'scale_cuts': scale_cuts # Minimum and maximum ell per probe
    }


# ------------------------------------------------------------------
# Step 3: Instantiate the dataset
# ------------------------------------------------------------------
# We select the shear-shear (SHE, SHE) auto-correlation as the
# observable defining the weak lensing probe.

data_WL = build_data_WL(('SHE', 'SHE', 1, 1), covmat_WL)
settings_WL = build_settings()


# ------------------------------------------------------------------
# Step 4: Initialise the likelihood
# ------------------------------------------------------------------
# The likelihood combines:
# - Background cosmology from cloelib (CAMBBackground)
# - Linear perturbations from cloelib (HMemuLinearPerturbations)
# - Non-linear corrections from cloelib (HMemuNonLinearPerturbations)
#
# Initialisation includes precomputations and typically dominates
# the one-time setup cost as it also prepares the data

like_WL = EuclidLikelihood_WL(
    data=data_WL,
    settings=settings_WL,
    Background=CAMBBackground,
    LinPerturbations=HMemuLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
    mode='coupled'
)


# ------------------------------------------------------------------
# Step 5: Define fiducial cosmological and nuisance parameters
# ------------------------------------------------------------------
# Evaluating the likelihood at this point should provide a non-zero value

default_pars = {
    # Cosmological parameters
    'H0': 70,
    'Omega_cdm0': 0.25,
    'Omega_b0': 0.05,
    'ns': 0.96,
    'As': 2e-9,
    'w0': -1,
    'wa': 0,
    'Omega_k0': 0,
    'mnu': 0.06,
    'gamma_MG': 0.545,
    'N_mnu': 1,
    'alpha_s': 0.0,
    'log10TAGN': 7.8,

    # Intrinsic alignment model
    'AIA': 1.72,
    'CIA': 0.0134,
    'EtaIA': -0.41,

    # Shear calibration systematics
    'multiplicative_bias_1': 0.0,
    'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0,
    'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0,
    'multiplicative_bias_6': 0.0,

    # Photometric redshift uncertainties (shear)
    'dz_shear_1': 0.0,
    'dz_shear_2': 0.0,
    'dz_shear_3': 0.0,
    'dz_shear_4': 0.0,
    'dz_shear_5': 0.0,
    'dz_shear_6': 0.0,

    # Photometric redshift uncertainties (shear)
    'width_shear_1': 0.0,
    'width_shear_2': 0.0,
    'width_shear_3': 0.0,
    'width_shear_4': 0.0,
    'width_shear_5': 0.0,
    'width_shear_6': 0.0,
}


# ------------------------------------------------------------------
# Step 6: Evaluate the log-likelihood
# ------------------------------------------------------------------
# This step computes the agreement between the theoretical prediction
# (given the parameters above) and the observed data vector.


loglike = like_WL.loglike(default_pars)
```

## Spectroscopic galaxy clustering full-shape

In this example, we compute a log-likelihood evaluation for spectroscopic galaxy clustering full-shape. In v0.3, the likelihood ingests the \texttt{euclidlib} data products directly: the fiducial cosmology is read from the data vector, and the conversion from $h^{-1}\,$Mpc to Mpc units and the assembly of the covariance blocks are handled internally.

```python
# ------------------------------------------------------------------
# Step 1: Define the spectroscopic redshift bins
# ------------------------------------------------------------------
# These labels identify the redshift bins used when reading the
# corresponding Euclid data-vector, covariance, and mixing-matrix files.

redshifts = [1.0, 1.2, 1.4, 1.65]
labels = [str(z).strip('0') for z in redshifts]


# ------------------------------------------------------------------
# Step 2: Read the spectroscopic GC data products
# ------------------------------------------------------------------
# euclidlib allows multiple files to be read at once by passing the
# redshift-bin labels to the reader functions.

datavec = power_spectrum_multipoles(
    'mps_pk_GCspectro_comet_EFT_z{}.fits',
    *labels
)

covariance = power_spectrum_multipole_covariance(
    'cov_pk_Gauss_GCspectro_comet_EFT_z{}_2500deg2.fits',
    *labels
)

mixing = power_spectrum_multipole_mixing_matrix(
    'mixmat_pk_GCspectro_identity_z{}.fits',
    *labels
)


# ------------------------------------------------------------------
# Step 3: Construct the data dictionary expected by cloelike
# ------------------------------------------------------------------
# For each redshift bin, we hand over the raw euclidlib objects:
# - Power spectrum multipoles (datavec), including the fiducial
#   cosmology needed for AP distortions
# - Covariance matrix (covariance)
# - Mixing matrix (mixing), optional

data = {
    'GCspectro': {
        z: {
            'datavec': datavec[('SPE', 'SPE', ii, ii)],
            'covariance': covariance[('SPE', 'SPE', ii, ii)],
            'mixing': mixing[('SPE', 'SPE', ii, ii)],
        }
        for ii, z in enumerate(labels)
    }
}


# ------------------------------------------------------------------
# Step 4: Define the analysis settings
# ------------------------------------------------------------------
# Scale cuts are specified for each redshift bin and multipole in
# h/Mpc units, and converted to 1/Mpc units to remain consistent
# with cloelib.

fid_h = datavec[('SPE', 'SPE', 0, 0)].fiducial_cosmology['H0'] / 100.0

scale_cuts = {
    '1.':   {'ell0': [0.0, 0.20], 'ell2': [0.0, 0.15], 'ell4': [0.0, 0.15]},
    '1.2':  {'ell0': [0.0, 0.25], 'ell2': [0.0, 0.20], 'ell4': [0.0, 0.20]},
    '1.4':  {'ell0': [0.0, 0.25], 'ell2': [0.0, 0.20], 'ell4': [0.0, 0.20]},
    '1.65': {'ell0': [0.0, 0.30], 'ell2': [0.0, 0.25], 'ell4': [0.0, 0.25]},
}

settings = {
    'GCspectro': {
        'scale_cuts': {
            z: {ell: [v * fid_h for v in cut] for ell, cut in cuts.items()}
            for z, cuts in scale_cuts.items()
        }
    }
}


# ------------------------------------------------------------------
# Step 5: Initialise the likelihood
# ------------------------------------------------------------------
# The likelihood combines:
# - Background cosmology from cloelib (CAMBBackground)
# - Spectroscopic galaxy clustering predictions from COMET
#   (CometEFT_SpectroPower)
#
# Optional arguments: AM_priors enables analytical marginalisation
# over the linear parameters, and num_mocks applies the Hartlap
# correction for covariances estimated from mocks.

like_spec = EuclidLikelihood_GCspectro_Pls(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    SpectroPower=CometEFT_SpectroPower,
)


# ------------------------------------------------------------------
# Step 6: Define cosmological and nuisance parameters
# ------------------------------------------------------------------
# The nuisance parameters are provided as arrays, with one value per
# spectroscopic redshift bin.

parameters = {
    # Cosmological parameters
    'H0': 67.0,
    'Omega_cdm0': 0.27,
    'Omega_b0': 0.049,
    'Omega_k0': 0.0,
    'mnu': 0.0,
    'N_mnu': 0,
    'w0': -1.0,
    'wa': 0.0,
    'ns': 0.96,
    'As': 2.1e-9,
    'alpha_s': 0.0,
    'gamma_MG': 0.545,

    # Galaxy bias and EFT nuisance parameters
    'b1': np.array([1.412, 1.769, 2.039, 2.496]),
    'b2': np.array([0.695, 0.870, 1.162, 2.010]),
    'bG2': np.array([-0.156, -0.299, -0.400, -0.555]),
    'bGam3': np.array([0.323, 0.621, 0.827, 1.137]),
    'c0': np.array([30.948, 37.116, 36.738, 53.627]),
    'c2': np.array([46.233, 53.071, 48.626, 60.962]),
    'c4': np.array([10.057, 10.385, 8.643, 8.711]),
    'cnlo': np.array([0.0, 0.0, 0.0, 0.0]),

    # Shot-noise and observational nuisance parameters
    'NP0': np.array([1.056, 1.152, 1.144, 1.309]),
    'NP20': np.array([0.0, 0.0, 0.0, 0.0]),
    'NP22': np.array([0.0, 0.0, 0.0, 0.0]),
    'fout': np.array([0.0, 0.0, 0.0, 0.0]),
    'sigmaz': np.array([0.0, 0.0, 0.0, 0.0]),
}


# ------------------------------------------------------------------
# Step 7: Evaluate the log-likelihood
# ------------------------------------------------------------------
# This computes the agreement between the spectroscopic GC data vector
# and the theoretical prediction for the parameter values above.

loglike = like_spec.loglike(parameters)
```

# Documentation

Comprehensive documentation for \texttt{cloelike} is available at \href{https://cloe-org.github.io/cloelike/dev/home/}{cloe-org.github.io/cloelike/dev/home/}. The documentation includes detailed API references, installation instructions, explanations about the software structure, and guides for integrating \texttt{cloelike} into your analysis workflows.

For practical examples, example scripts, and interactive tutorials, visit the \href{https://github.com/cloe-org/playground}{cloe-org/playground} repository, which hosts a collection of Jupyter notebooks showcasing typical use cases and advanced features. In particular, \href{https://github.com/cloe-org/playground/tree/main/tutorials/likelihood}{cloe-org/playground/tutorials/likelihood} shows examples on how to run single $\chi^2$ evaluations.

\texttt{cloelike} is prepared to interface with photometric and spectroscopic observables data formats compliant with \texttt{euclidlib}[^euclidlib-docs] formats, using \texttt{cosmolib}[^cosmolib] dataclasses.

[^euclidlib-docs]: <https://euclidlib.readthedocs.io/en/latest/>

[^cosmolib]: <https://github.com/astro-ph/cosmolib>

# Availability

- **Source:** [github.com/cloe-org/cloelike](https://github.com/cloe-org/cloelike)
- **Version:** 0.3
- **License:** MIT
- **Install (PyPI):** \texttt{pip install cloelike} (also installs a compatible \texttt{cloelib} $\geq$ 0.11)
- **Documentation:** [https://cloe-org.github.io/cloelike/dev/home/](https://cloe-org.github.io/cloelike/dev/home/)
- **Examples:** [github.com/cloe-org/playground](https://github.com/cloe-org/playground)

# Acknowledgements

We thank the broader CLOE software development team for foundational work that motivated this library. The scientific development of cloe-org is coordinated through the Joint Cosmology Key Project one (DR1-KP-JC-1) of the Euclid Consortium, led by C. Carbone, M. Crocce, and I. Tutusaus. In this context, \texttt{cloe-org} has been adopted as the default analysis code of the Euclid Consortium. G.C.H. acknowledges that this project is part of the project UNICORN with file number VI.Veni.242.110 of the research programme Talent Programme Veni Science domain 2024 which is (partly) financed by the Dutch Research Council (NWO) under the grant [https://doi.org/10.61686/ZCPQI32997](https://doi.org/10.61686/ZCPQI32997). M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). C.M. is supported by the Agenzia Spaziale Italiana project "Attività scientifica per la missione Euclid – fase E ACCORDO ATTUATIVO n. 2024-10-HH.0.". I.T. acknowledges support from the Spanish Ministerio de Ciencia, Innovación y Universidades, projects PID2022-141079NB, PID2022-138896NB; the European Research Executive Agency HORIZON-MSCA-2021-SE-01 Research and Innovation programme under the Marie Skłodowska-Curie grant agreement number 101086388 (LACEGAL) and the programme Unidad de Excelencia María de Maeztu, project CEX2020-001058-M. N.G. acknowledges the support of the Royal Society as a Newton International Fellow (NIF\textbackslash R1\textbackslash 252792).

We acknowledge EuroHPC Joint Undertaking for awarding the project ID EHPC-EXT-2024E02-083 access to Leonardo hosted by CINECA, Italy. We acknowledge the use of Spanish Supercomputing Network (RES) resources provided by the Barcelona Supercomputing Center (BSC) in MareNostrum 5 under allocations AECT-2024-3-0020, 2025-1-0045, 2025-2-0046, 2025-3-0036. We acknowledge support from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program with Grant agreement No. 101053992 for computational resources.

The Euclid Consortium acknowledges the European Space Agency and a number of agencies and institutes that have supported its development, in particular: the Agenzia Spaziale Italiana; the Austrian Forschungsförderungsgesellschaft funded through BMIMI; the Belgian Science Policy; the Canadian Euclid Consortium; the Deutsches Zentrum für Luft- und Raumfahrt; DTU Space and the Niels Bohr Institute (Denmark); the French Centre National d'Études Spatiales; the Fundação para a Ciência e a Tecnologia; the Hungarian Academy of Sciences; the Ministerio de Ciencia, Innovación y Universidades; the National Aeronautics and Space Administration; the National Astronomical Observatory of Japan; the Netherlands Research School for Astronomy; the Norwegian Space Agency; the Research Council of Finland; the Romanian Space Agency; the Swiss Space Office (SSO) at the State Secretariat for Education, Research, and Innovation (SERI); and the United Kingdom Space Agency. A complete and detailed list is available at [www.euclid-ec.org/consortium/community/](https://www.euclid-ec.org/consortium/community/).

# Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors (Bonici, Cañas-Herrera, Carrilho, Casas, Moretti, Pezzotta) are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of \texttt{cloelike}, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: Review of software and design.
- **G. Cañas-Herrera**: Repository set-up and overall software architecture; implementation of `PhotoLikelihoodBase` including data ingestion, scale-cut masking, covariance inversion, geometric rebinning of $C_\ell$ data vectors, and `lru_cache` optimisation; mixin composition framework; CI pipeline configuration; pre-commit and code-quality tooling; issue and pull-request templates; README, documentation infrastructure, and community contribution tracking (\texttt{all-contributors}); \texttt{pyproject.toml} versioning and release workflows; BAO unit tests. Resources, writing - original draft, visualization, project administration.
- **P. Carrilho**: Implementation of systematic nuisance parameters for photometric tracers (multiplicative shear bias, photometric redshift shifts and width parameters); coupled/uncoupled mode flag for pseudo-$C_\ell$ likelihood with mixing matrices; photometric likelihood refinements and notebook tutorials.
- **S. Casas**: Review of design and ideas.
- **C. Moretti**: Implementation of `EuclidLikelihood_BAO` (BAO $\alpha$-parameter likelihood); spectroscopic likelihood bug fixes; unit tests for BAO likelihood; repository housekeeping.
- **A. Pezzotta**: Core implementation of `EuclidLikelihood_GCspectro_Pls` (full-shape spectroscopic power spectrum multipoles); analytical marginalisation over linear bias parameters; spectroscopic data-vector and covariance-matrix handling; unit tests for spectroscopic likelihood; homogenisation of photometric and spectroscopic likelihood APIs.

C. Carbone, M. Crocce, and I. Tutusaus: Project administration, Supervision, Validation. The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot, following the specification of the same name. A full, categorised breakdown of each contributor's role—including code, documentation, testing, ideas, project management, and more—is available in the \texttt{README} of the \texttt{cloelike} repository, fully detailed within the \texttt{cloelike} docs.

# References
