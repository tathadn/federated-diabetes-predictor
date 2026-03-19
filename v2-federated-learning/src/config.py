"""Configuration settings for the V2 Federated Learning project."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEDERATED_DATA_DIR = DATA_DIR / "federated"
SPLITS_DIR = DATA_DIR / "splits"

# ---------------------------------------------------------------------------
# Results directories
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
REPORTS_DIR = RESULTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Source dataset (symlinked / copied from V1)
# ---------------------------------------------------------------------------
RAW_DATASET_FILENAME = "diabetes_binary.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME

# V1 raw data path (used by download_dataset.py as fallback source)
V1_RAW_DATA_PATH = (
    PROJECT_ROOT.parent / "v1-basic-ml" / "data" / "raw" / RAW_DATASET_FILENAME
)

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Diabetes_binary"

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Federated learning settings
# ---------------------------------------------------------------------------
N_CLIENTS: int = 3
NUM_ROUNDS: int = 15
LOCAL_EPOCHS: int = 1
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.01

# Dirichlet alpha for non-IID distribution (lower = more heterogeneous)
DIRICHLET_ALPHA: float = 0.5

# Server address for distributed mode
SERVER_ADDRESS: str = "127.0.0.1:8080"

# Fraction of clients sampled per round
FRACTION_FIT: float = 1.0
FRACTION_EVALUATE: float = 1.0
MIN_FIT_CLIENTS: int = 2
MIN_EVALUATE_CLIENTS: int = 2
MIN_AVAILABLE_CLIENTS: int = 2

# FedProx mu (proximal term)
FEDPROX_MU: float = 0.1

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
MODEL_TYPE: str = "logistic"  # "logistic" | "random_forest"

LOGISTIC_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "warm_start": True,
}

# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------
CONVERGENCE_ROUNDS_TARGET: int = 20  # reach stable accuracy within N rounds
ACCURACY_LOSS_TOLERANCE: float = 0.05  # V2 accuracy >= V1 - 5%
