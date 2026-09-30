# Spectroscopic Galaxy Clustering Likelihoods

These classes implement Gaussian likelihoods for Euclid spectroscopic galaxy clustering probes, covering power spectrum multipoles ($P_\ell$), correlation function multipoles ($\xi_\ell$), Baryon Acoustic Oscillations (BAO), and their joint combination. For all available modules, if a numerical covariance is employed in the fit, we correct the inverse covariance with the Hartlap factor, based on the provided number of independent realisations.

---

## Power Spectrum Multipoles

### EuclidLikelihood_GCspectro_Pls

::: cloelike.EuclidLikelihood_GCspectro_Pls
options:
show_source: true

---

## Correlation Function Multipoles

### EuclidLikelihood_GCspectro_xils

::: cloelike.EuclidLikelihood_GCspectro_xils
options:
show_source: true

---

## Baryon Acoustic Oscillations (BAO)

### EuclidLikelihood_GCspectro_BAO

::: cloelike.EuclidLikelihood_GCspectro_BAO
options:
show_source: true

---

## Joint Power Spectrum Multipoles + BAO

### EuclidLikelihood_GCspectro_Pls_BAO

::: cloelike.EuclidLikelihood_GCspectro_Pls_BAO
options:
show_source: true
