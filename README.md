# SINDy Feature Extensions

This work was realized in 2019 and was designed to extend the original pySINDy repo: [https://github.com/luckystarufo/pySINDy](https://github.com/luckystarufo/pySINDy) version 0.2.0.
The original repository is now deprecated and the updated version can be found here: [PySINDy](https://github.com/dynamicslab/pysindy)

This repository provides a collection of modular, reusable extensions for the original pySINDy library, designed to enhance sparse system identification. These tools allow for the incorporation of domain knowledge through custom feature libraries and provide robust methods for differentiation and model selection.

The core components include:
- **Feature Libraries**: `TrigLibrary`, `ChebyshevLibrary`, and `TimeDelayLibrary` for creating domain-specific candidate functions.
- **Derivative Estimators**: Robust options like `SavitzkyGolay` and `FiniteDiff` for handling noisy data.
- **Sparse Regression**: A cross-validated Sequentially Thresholded Least Squares (`cv_stlsq`) optimizer for automated hyperparameter tuning.
- **Utilities**: Helpers for time-series validation, including leakage detection and safe lag generation.

All components are designed to be compatible with the PySINDy API (`fit`/`transform`/`get_feature_names`) but can also be used independently.

## Installation

First, clone the repository and set up a virtual environment.

```bash
git clone https://github.com/emagnon/sindy-feature-extensions.git
cd sindy-feature-extensions
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
. \.venv\Scripts\Activate.ps1
```

Install the package and its dependencies. The `[examples]` option includes `pysindy` and `matplotlib`.

```bash
# Install the package with dependencies for the example
pip install -e ".[examples]"

# For development (testing, formatting)
pip install -e ".[dev]"
```

## Running the Example

To run the damped pendulum simulation and see the results, execute the following command:

```bash
python examples/pendulum_trig_vs_poly.py
```

## Running Tests

To ensure all components are working correctly, run the test suite using `pytest`.

```bash
pytest -q -rA
```
