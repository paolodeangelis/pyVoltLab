# ⚡ Voltage Profile Generator

🚧 Under active development — this is not the final version. Further improvements, optimizations, and updates are on the way.

This repository provides the implementation for generating voltage profiles as presented in the preprint: 

[![arXiv](https://img.shields.io/badge/arXiv-2510.05020-b31b1b.svg)](https://doi.org/10.48550/arXiv.2511.22504)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2510.05020-blue)](https://doi.org/10.48550/arXiv.2511.22504)

**Screening novel cathode materials from the Energy-GNoME database using MACE machine learning force field and DFT.**
Nada Alghamdi, Paolo de Angelis, Pietro Asinari, Eliodoro Chiavazzo
## Installation
It is recommended to use a virtual environment (**venv**) or **Conda** to manage dependencies.
Ensure you have Python installed. Navigate to the root folder (where `pyproject.toml` is located) and run:

```bash
pip install .
```

## Usage

To generate the voltage profile for the **LiCoO2** example provided in the repository:

### 1. Navigate to the example directory

```bash
cd examples/LiCoO2
```

### 2. Run the simulation

Execute the script using the provided configuration:

```
python3 ../../build/lib/mc_mace/pymc.py zerok-voltage settings.yaml

```

# Copyright

Copyright (c) 2024-2026, Paolo De Angelis, Nada Alghamdi


# Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
