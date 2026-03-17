# cloelike – The likelihood module for the Cosmology Likelihood for Observables in Euclid project

**cloelike** is the likelihood module for _Euclid_ primary observables, interfacing with `cloelib`

We welcome feedback from the **Euclid community** and beyond to refine and improve this module!

[![CI](https://github.com/cloe-org/cloelike/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cloe-org/cloelike/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-blue?logo=pytest)](https://docs.pytest.org/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-purple?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Code Style: Prettier](https://img.shields.io/badge/code%20style-prettier-ff69b4.svg?logo=prettier&logoColor=white)](https://prettier.io/)

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🚀 Installation](#-installation)
- [📊 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

🔹 **Intuitive & User-Friendly** – General description of the likelihood options by classes

🔹 **loglike calculation for Euclid primary probes** – currently supporting 3x2pt and spectroscopic galaxy clustering full-shape

---

## 🚀 Installation

To install `cloelike` source code, clone the repository and install it via `pip`:

```sh
pip install .
```

To work with the latest stable release of the code, move to the latest tag by typing:

```sh
git checkout name-latest-release
```

with name-latest-release the latest name that appears in "Releases".

---

## 📂 Dependency: `cloelib`

`cloelike` depends on the [`cloelib`](https://github.com/cloe-org/cloelib) package (private for now).

For the time being, please install `cloelib` manually by following the installation instructions provided in its [GitHub repository](https://github.com/cloe-org/cloelib).

---

## 📊 Usage

Explore the **tutorials** in the `cloe-org/playground` repository for examples on how to compute cosmological observables and other key quantities!

---

## 🤝 Contributing

Please review the organization's general contribution guidelines and the specific guidelines for this repository in the [CONTRIBUTING.md](CONTRIBUTING.md) file. Once you're familiar with the guidelines, follow these steps:

1️⃣ Create a new branch:

```sh
git checkout -b issue-<number>-<short-descriptive-title>
```

2️⃣ Implement your changes following project style guidelines.

3️⃣ Commit your modifications:

```sh
git commit -m "Add feature: [brief description]"
```

4️⃣ Push your branch:

```sh
git push origin feature/your-feature-name
```

5️⃣ Open a **pull request** and contribute to the project!

---

## 📜 License

This project is licensed under the **MIT** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

👩‍💻🧑‍💻 Authored by M. Bonici, G. Cañas-Herrera, P. Carrilho, S. Casas, C. Moretti, and A. Pezzotta (listed in alphabetical order).

## 🤝 Contributors

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-gcanasherrera" title="Bug reports">🐛</a> <a href="#content-gcanasherrera" title="Content">🖋</a> <a href="#data-gcanasherrera" title="Data">🔣</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#question-gcanasherrera" title="Answering Questions">💬</a> <a href="#test-gcanasherrera" title="Tests">⚠️</a> <a href="#talk-gcanasherrera" title="Talks">📢</a> <a href="#review-gcanasherrera" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#ideas-AndreaPezzotta" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-AndreaPezzotta" title="Bug reports">🐛</a> <a href="#content-AndreaPezzotta" title="Content">🖋</a> <a href="#data-AndreaPezzotta" title="Data">🔣</a> <a href="#doc-AndreaPezzotta" title="Documentation">📖</a> <a href="#talk-AndreaPezzotta" title="Talks">📢</a> <a href="#review-AndreaPezzotta" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-PedroCarrilho" title="Bug reports">🐛</a> <a href="#content-PedroCarrilho" title="Content">🖋</a> <a href="#data-PedroCarrilho" title="Data">🔣</a> <a href="#doc-PedroCarrilho" title="Documentation">📖</a> <a href="#talk-PedroCarrilho" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://marcobonici.github.io/"><img src="https://avatars.githubusercontent.com/u/58727599?v=4?s=100" width="100px;" alt="Marco Bonici"/><br /><sub><b>Marco Bonici</b></sub></a><br /><a href="#review-marcobonici" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#ideas-santiagocasas" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-santiagocasas" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chiaramoretti"><img src="https://avatars.githubusercontent.com/u/12472732?v=4?s=100" width="100px;" alt="Chiara Moretti"/><br /><sub><b>Chiara Moretti</b></sub></a><br /><a href="#code-chiaramoretti" title="Code">💻</a> <a href="#ideas-chiaramoretti" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-chiaramoretti" title="Maintenance">🚧</a> <a href="#content-chiaramoretti" title="Content">🖋</a> <a href="#data-chiaramoretti" title="Data">🔣</a> <a href="#talk-chiaramoretti" title="Talks">📢</a> <a href="#bug-chiaramoretti" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://ntessore.page"><img src="https://avatars.githubusercontent.com/u/3993688?v=4?s=100" width="100px;" alt="Nicolas Tessore"/><br /><sub><b>Nicolas Tessore</b></sub></a><br /><a href="#review-ntessore" title="Reviewed Pull Requests">👀</a> <a href="#ideas-ntessore" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/davidesciotti"><img src="https://avatars.githubusercontent.com/u/84071067?v=4?s=100" width="100px;" alt="Davide Sciotti"/><br /><sub><b>Davide Sciotti</b></sub></a><br /><a href="#code-davidesciotti" title="Code">💻</a> <a href="#bug-davidesciotti" title="Bug reports">🐛</a> <a href="#ideas-davidesciotti" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zahrabaghkhani"><img src="https://avatars.githubusercontent.com/u/47903409?v=4?s=100" width="100px;" alt="Zahra Baghkhani"/><br /><sub><b>Zahra Baghkhani</b></sub></a><br /><a href="#code-zahrabaghkhani" title="Code">💻</a> <a href="#doc-zahrabaghkhani" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jipdebuck"><img src="https://avatars.githubusercontent.com/u/236796982?v=4?s=100" width="100px;" alt="Jip de Buck"/><br /><sub><b>Jip de Buck</b></sub></a><br /><a href="#bug-jipdebuck" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
