# DIINA: Dynamic Inhibitory Neural Architecture for Multilingual Processing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20612090.svg)](https://doi.org/10.5281/zenodo.20612090)
[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/Pegi1727/DIINA-/releases/tag/v1.0.0)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--6235--3010-green.svg)](https://orcid.org/0009-0000-6235-3010)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation and reproducibility telemetry for **DIINA (Dynamic Inhibitory Neural Architecture)**. DIINA couples an ultra-fast local convolutional pathway with an adaptive selective state-space / attention slow pathway via a biologically inspired **Linguistic Inhibition Gate (LIG)** to resolve long-range dependencies, syntactic licensing, and lexical ambiguities in low-resource and morphologically complex linguistic domains.

---

## Repository Tree Structure
```text
DIINA-/
├── CITATION.cff                      # GitHub-native citation metadata
├── LICENSE                           # MIT License
├── README.md                         # Project documentation and empirical benchmarks
├── figures/                          # Publication-ready high-resolution figures
│   ├── figure_02_biological_mapping.png
│   ├── figure_03_gating_logic.png
│   ├── figure_04_token_flow.png
│   ├── figure_05_accuracy_vs_complexity.png
│   ├── figure_06_inhibition_vs_hardness.png
│   ├── figure_07_compute_cost.png
│   └── figure_09_convergence.png
├── models/                           # Core neural architecture implementations
│   ├── diina.py                      # Complete DIINA module: Fast Path + Slow Path + LIG
│   ├── mamba.py                      # Selective State Space (SSM) abstract interface
│   └── DIINA_15.py                   # Extended modular dual-path prototype
├── reproducibility/                  # Empirical verification suites & notebooks
│   ├── R/                            # Statistical verification & gating dynamics scripts
│   ├── notebooks/                    # Interactive diagnostic Jupyter notebooks
│   ├── summary.json                  # Aggregated benchmark telemetry
│   └── raw_results.csv               # Raw multi-seed inference logs
└── data/                             # Dataset schemas and diagnostic loaders
└── data_engine.py                # Synthetic & empirical task generator
