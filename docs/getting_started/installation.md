# Installation

## Requirements

- Python ≥ 3.12 (< 3.13 due to upstream dependency constraints)
- [`cloelib`](https://github.com/cloe-org/cloelib) ≥ 0.11, installed automatically as a dependency

## Installing cloelike

### From PyPI (latest release)

```sh
pip install cloelike
```

This also installs [`cloelib`](https://pypi.org/project/cloelib/) from PyPI. To use the development version of `cloelib` instead, install it first following the instructions in its [GitHub repository](https://github.com/cloe-org/cloelib):

```sh
pip install git+https://github.com/cloe-org/cloelib.git
```

### From source (latest development version)

```sh
git clone https://github.com/cloe-org/cloelike.git
cd cloelike
pip install .
```

### From a specific release

To work with the latest stable release, check out the corresponding tag:

```sh
git checkout <name-latest-release>
pip install .
```

where `<name-latest-release>` is the tag name visible under [Releases](https://github.com/cloe-org/cloelike/releases).

## Optional dependencies

All dependencies of external codes can be installed as extras from `cloelib`. Check out `cloelib` docs.
