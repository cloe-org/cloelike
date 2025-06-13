# cloelike – The likelihood module for the Cosmology Likelihood for Observables in Euclid project

**cloelike** is the likelihood module for _Euclid_ primary observables, interfacing with `cloelib`

We welcome feedback from the **Euclid community** and beyond to refine and improve this module!  

[![CI](https://github.com/cloe-org/cloelike/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/cloe-org/cloelike/actions/workflows/ci.yaml)

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
   git checkout -b feature/your-feature-name
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
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-gcanasherrera" title="Bug reports">🐛</a> <a href="#content-gcanasherrera" title="Content">🖋</a> <a href="#data-gcanasherrera" title="Data">🔣</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#question-gcanasherrera" title="Answering Questions">💬</a> <a href="#test-gcanasherrera" title="Tests">⚠️</a> <a href="#talk-gcanasherrera" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#ideas-AndreaPezzotta" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-AndreaPezzotta" title="Bug reports">🐛</a> <a href="#content-AndreaPezzotta" title="Content">🖋</a> <a href="#data-AndreaPezzotta" title="Data">🔣</a> <a href="#doc-AndreaPezzotta" title="Documentation">📖</a> <a href="#talk-AndreaPezzotta" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-PedroCarrilho" title="Bug reports">🐛</a> <a href="#content-PedroCarrilho" title="Content">🖋</a> <a href="#data-PedroCarrilho" title="Data">🔣</a> <a href="#doc-PedroCarrilho" title="Documentation">📖</a> <a href="#talk-PedroCarrilho" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
