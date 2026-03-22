# V2: Federated Learning for Diabetes Prediction

An extension of the V1 centralized baseline that implements **privacy-preserving federated learning** using the [Flower](https://flower.ai) framework. Three simulated healthcare organizations collaboratively train a global model without ever sharing raw patient data.

## Highlights

| | Centralized (V1) | Federated (V2) |
|---|---|---|
| Data location | Single server | Stays on each client |
| Privacy | None | High (weights only) |
| Communication | N/A | ~KB total (LR model) |
| Scalability | Requires data movement | Scales to any # clients |
| HIPAA / GDPR | Problematic | Aligned |

## Quick Start

### 1. Install dependencies

```bash
cd v2-federated-learning
pip install -e ".[dev,notebook]"
```

### 2. Download / copy dataset

```bash
python scripts/download_dataset.py
```

This copies `diabetes_binary.csv` from V1 automatically if it exists.

### 3. Create non-IID data splits

Run `notebooks/01_data_distribution.ipynb` or:

```python
from src.data_utils import load_dataset, create_noniid_splits, save_client_data
save_client_data(create_noniid_splits(load_dataset()))
```

### 4. Run federated training

```bash
# Single-machine simulation (recommended)
python scripts/run_federated_training.py

# With custom settings
python scripts/run_federated_training.py --rounds 20 --strategy fedprox

# Distributed (3 terminals)
python scripts/run_federated_training.py --mode distributed
python src/client.py 0
python src/client.py 1
python src/client.py 2
```

### 5. Analyse results

```bash
python scripts/visualize_results.py
```

Or run the notebooks in order:
- `notebooks/01_data_distribution.ipynb`
- `notebooks/02_federated_training.ipynb`
- `notebooks/03_federated_evaluation.ipynb`
- `notebooks/04_comparison_analysis.ipynb`

### 6. Run tests

```bash
pytest tests/ -v --cov=src
```

## Project Structure

```
v2-federated-learning/
├── notebooks/          # Step-by-step analysis notebooks
├── src/
│   ├── config.py       # All hyperparameters and paths
│   ├── data_utils.py   # Non-IID Dirichlet splitting
│   ├── models.py       # Model factories + FL param helpers
│   ├── client.py       # Flower NumPyClient implementation
│   ├── server.py       # Flower server + strategy builder
│   ├── strategies.py   # LoggingFedAvg + FedProxStrategy
│   ├── metrics.py      # Evaluation + communication cost
│   └── privacy.py      # Gaussian DP utilities
├── scripts/
│   ├── run_federated_training.py
│   ├── run_single_machine_simulation.py
│   ├── visualize_results.py
│   └── download_dataset.py
├── tests/              # Pytest test suite (>80% coverage)
├── configs/            # JSON configuration files
├── data/               # Raw + federated client splits
└── results/            # Plots, models, logs, reports
```

## Key Concepts

### FedAvg (Federated Averaging)
Each client trains locally for E epochs, then the server computes a weighted average:

```
w_global = Σ (n_i / n) * w_i
```

### Non-IID Data Distribution
Uses the **Dirichlet distribution** (parameter `α`) to assign different class proportions to each client. Lower `α` = more heterogeneous.

### Privacy Properties
- Raw data **never leaves** the client
- Only model weights are transmitted
- Optional **differential privacy** via Gaussian mechanism (`src/privacy.py`)

## Configuration

Edit `configs/default_config.json` or override via `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_clients` | 3 | Number of federated clients |
| `num_rounds` | 15 | FL communication rounds |
| `local_epochs` | 1 | Local training epochs per round |
| `dirichlet_alpha` | 0.5 | Non-IID heterogeneity |
| `strategy` | fedavg | `fedavg` or `fedprox` |

## Results

After training, results are saved in `results/`:
- `results/models/global_model.pkl` — final global model
- `results/models/fl_metrics_history.json` — per-round metrics
- `results/plots/` — convergence curves, comparisons
- `results/reports/` — V1 vs V2 comparison JSON + CSV

## Tools Used

- [Claude](https://claude.ai) (Anthropic) — AI assistant used during development
