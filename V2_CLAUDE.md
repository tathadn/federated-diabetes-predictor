# Claude.md - Version 2: Diabetes Prediction - Federated Learning Extension

## Project Overview
This is an advanced machine learning project that extends Version 1 by implementing federated learning. The project demonstrates privacy-preserving machine learning where multiple clients train models locally and a central server aggregates the updates without accessing raw data.

## Project Goals
- Implement federated learning using the Flower framework
- Simulate distributed healthcare organizations training collaboratively
- Demonstrate privacy-preserving ML techniques
- Compare federated vs centralized learning approaches
- Showcase understanding of distributed systems and privacy
- Create production-ready federated learning pipeline

## Technology Stack
- **Language**: Python 3.8+
- **Core ML Framework**: Scikit-learn, TensorFlow/Keras
- **Federated Learning**: Flower (flwr)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Network**: Sockets (or gRPC via Flower)
- **Version Control**: Git, GitHub
- **Testing**: Pytest

## Key Federated Learning Concepts

### Federated Averaging (FedAvg)
- Algorithm for aggregating model updates from multiple clients
- Each client trains locally for E epochs
- Sends updated weights to server
- Server computes weighted average of client updates
- Formula: `w_global = Σ(n_i / n) * w_i` where n_i is client data size

### Non-IID (Heterogeneous) Data Distribution
- Data at each client has different distribution
- Simulates real-world scenarios (different hospitals, regions)
- More challenging than IID distribution
- Use Dirichlet distribution to create realistic scenarios
- Document data statistics for each client

### Privacy-Preserving Properties
- Raw data never leaves client devices
- Only model updates transmitted
- Differential privacy can add noise for stronger guarantees
- Suitable for regulated industries (healthcare, finance)

### Communication Efficiency
- Focus on reducing number of communication rounds
- Each round involves upload and download
- Trade-off: accuracy vs communication cost
- Monitor bandwidth usage

## Project Structure
```
v2-federated-learning/
├── notebooks/
│   ├── 01_data_distribution.ipynb      # Non-IID data simulation
│   ├── 02_federated_training.ipynb     # FL training process
│   ├── 03_federated_evaluation.ipynb   # FL evaluation & metrics
│   └── 04_comparison_analysis.ipynb    # V1 vs V2 comparison
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                   # Federated data utilities
│   ├── client.py                       # Flower client implementation
│   ├── server.py                       # Flower server implementation
│   ├── models.py                       # Shared model definitions
│   ├── strategies.py                   # Custom aggregation strategies
│   ├── metrics.py                      # Federated metrics calculation
│   ├── privacy.py                      # Privacy utilities (optional)
│   └── config.py                       # Configuration settings
│
├── data/
│   ├── raw/                           # Original dataset
│   │   └── .gitkeep
│   ├── federated/                     # Client data splits
│   │   ├── client_0_data.csv
│   │   ├── client_1_data.csv
│   │   ├── client_2_data.csv
│   │   └── .gitkeep
│   └── splits/                        # Train/val/test splits
│       └── .gitkeep
│
├── results/
│   ├── plots/                         # Federated learning plots
│   │   ├── convergence_curves.png
│   │   ├── communication_analysis.png
│   │   ├── privacy_utility_tradeoff.png
│   │   └── .gitkeep
│   ├── models/                        # Global and client models
│   │   ├── global_model.pkl
│   │   ├── client_0_model.pkl
│   │   └── .gitkeep
│   ├── logs/                          # Training logs
│   │   └── .gitkeep
│   └── reports/                       # Analysis reports
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_client.py
│   └── test_server.py
│
├── configs/
│   ├── default_config.json            # Default FL configuration
│   └── tuning_config.json             # Hyperparameter tuning config
│
├── scripts/
│   ├── run_federated_training.py      # Main FL training script
│   ├── run_single_machine_simulation.py # Simulate all clients locally
│   ├── download_dataset.py            # Dataset download utility
│   └── visualize_results.py           # Results visualization script
│
├── requirements.txt                    # Project dependencies
├── setup.py                           # Package setup
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules
```

## Implementation Workflow

### Phase 1: Non-IID Data Distribution (1-2 hours)
**File**: `notebooks/01_data_distribution.ipynb`

**Objectives**:
- Load original dataset
- Create realistic non-IID distribution
- Split data among clients
- Validate distribution statistics
- Save client datasets

**Implementation Steps**:

1. **Load Original Dataset**:
```python
import pandas as pd
df = pd.read_csv('diabetes.csv')
print(f"Total records: {len(df)}")
print(f"Class distribution: {df['Diabetes_binary'].value_counts()}")
```

2. **Create Non-IID Distribution**:
```python
import numpy as np
from numpy.random import dirichlet

n_clients = 3
alpha = 0.1  # Lower = more heterogeneous

# Dirichlet distribution
np.random.seed(42)
proportions = dirichlet([alpha] * n_clients)

# Alternative: Create with class-specific bias
# Each client has different class distribution
```

3. **Distribute Data to Clients**:
- Client 0: ~30% of data
- Client 1: ~35% of data
- Client 2: ~35% of data
- Each with different class distributions

4. **Validation Checks**:
- Total samples across clients = original size
- No data overlap between clients
- Document class distribution per client
- Check feature statistics per client

5. **Save Client Data**:
```python
for client_id in range(n_clients):
    client_df = distribute_data(df, client_id)
    client_df.to_csv(f'client_{client_id}_data.csv', index=False)
    log_client_statistics(client_id, client_df)
```

**Expected Outputs**:
- Individual CSV files for each client
- Data distribution statistics
- Visualization of non-IID patterns
- Documentation of distribution rationale

**Key Metrics to Analyze**:
- Samples per client
- Class distribution per client
- Feature means and stds per client
- Coefficient of variation for features

---

### Phase 2: Flower Framework Setup & Client Implementation (2-3 hours)
**File**: `src/client.py` and related modules

**Objectives**:
- Set up Flower framework
- Implement federated client
- Handle local training
- Manage model updates

**Installation**:
```bash
pip install flwr[simulation]
# Or with TensorFlow support:
pip install flwr[tf]
```

**Client Implementation Strategy**:

1. **Define Flower Client Class**:
```python
import flwr as fl
from typing import Dict, Tuple, List
import numpy as np

class DiabetesClient(fl.client.NumPyClient):
    """Federated client for diabetes prediction."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as list of numpy arrays."""
        # Extract parameters from model
        # For sklearn: [coef_, intercept_]
        # For Keras: model.get_weights()
        
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple:
        """Train model locally and return updated parameters."""
        # Update model with received parameters
        # Train for local_epochs
        # Return updated parameters, sample count, metrics
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple:
        """Evaluate model on local test set."""
        # Update model with parameters
        # Compute loss and accuracy
        # Return loss, samples, metrics
```

2. **Key Client Methods**:

**get_parameters()**:
- Extract weights from local model
- For sklearn (Logistic Regression): [coef_, intercept_]
- For Keras: model.get_weights()
- Return as list of numpy arrays

**fit()**:
- Receive global model parameters
- Update local model weights
- Train for `local_epochs` (typically 1-5)
- Calculate local metrics
- Return: (updated_parameters, num_examples, metrics_dict)

**evaluate()**:
- Receive global model parameters
- Update local model
- Evaluate on local test set
- Return: (loss, num_examples, metrics_dict)

3. **Client Configuration**:
```python
client_config = {
    'local_epochs': 1,          # Epochs to train locally
    'batch_size': 32,           # Batch size for training
    'learning_rate': 0.01,      # Learning rate
    'patience': 10,             # Early stopping patience
}
```

**Expected Outputs**:
- Functional client.py module
- Client can receive/send model updates
- Logging of client-side metrics
- Error handling for edge cases

---

### Phase 3: Federated Server Implementation (2-3 hours)
**File**: `src/server.py`

**Objectives**:
- Initialize global model
- Implement model aggregation
- Manage training rounds
- Track convergence

**Server Implementation Strategy**:

1. **Server Setup**:
```python
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.server.client_manager import SimpleClientManager

# Initialize global model
global_model = initialize_model()
initial_params = extract_parameters(global_model)

# Create strategy
strategy = FedAvg(
    fraction_fit=1.0,           # Use all available clients
    fraction_evaluate=1.0,       # Evaluate all clients
    min_fit_clients=2,           # Minimum clients to fit
    min_evaluate_clients=2,      # Minimum clients to evaluate
    min_available_clients=2,     # Must have at least 2 clients
    initial_parameters=initial_params,
)
```

2. **Strategy Selection**:

**FedAvg** (Federated Averaging):
```
Best for: General purpose FL
Aggregation: w_t+1 = Σ(n_i/n) * w_i^t
Pros: Simple, standard approach
Cons: Sensitive to non-IID data
```

**FedProx** (Federated Proximal):
```
Best for: Heterogeneous environments
Aggregation: Adds regularization term
Pros: Better with non-IID data
Cons: Slightly more complex
```

3. **Server Loop**:
```python
# Start server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10),
)
```

4. **Custom Aggregation** (Optional):
```python
from flwr.server.strategy import Strategy

class CustomFedAvg(FedAvg):
    """Custom aggregation strategy with logging."""
    
    def aggregate_fit(self, server_round, results, failures):
        """Custom fit aggregation."""
        # Log metrics
        # Custom aggregation logic
        # Return aggregated weights
        
    def aggregate_evaluate(self, server_round, results, failures):
        """Custom evaluation aggregation."""
        # Process evaluation results
        # Log per-client metrics
        # Return aggregated metrics
```

**Expected Outputs**:
- Functional server.py module
- Proper model aggregation
- Convergence tracking
- Detailed logging of FL process

---

### Phase 4: Federated Training Execution (1-2 hours)
**File**: `scripts/run_federated_training.py`

**Objectives**:
- Execute federated training
- Monitor convergence
- Track metrics per round
- Save checkpoints

**Implementation Approaches**:

**Option 1: Single-Machine Simulation** (Recommended for testing):
```python
# Simulate federated training on single machine
import flwr as fl
from flwr.simulation import start_simulation

# Start simulation
start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

**Option 2: Distributed Setup** (Multiple machines):
```
Terminal 1: python server.py
Terminal 2: python client.py 0
Terminal 3: python client.py 1
Terminal 4: python client.py 2
```

**Training Script Structure**:
```python
def client_fn(cid: str):
    """Create client function for simulation."""
    # Load client's data
    X_train, y_train = load_client_data(cid)
    X_test, y_test = load_client_test_data(cid)
    
    # Initialize model
    model = initialize_model()
    
    # Return client
    return DiabetesClient(model, X_train, y_train, X_test, y_test, cid)

# Run simulation
start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources={'num_cpus': 1, 'num_gpus': 0},
)
```

**Monitoring & Logging**:
```python
# Track per-round metrics
metrics_history = {
    'round': [],
    'loss': [],
    'accuracy': [],
    'client_accuracies': []
}

# After each round:
metrics_history['round'].append(round_num)
metrics_history['loss'].append(aggregated_loss)
metrics_history['accuracy'].append(aggregated_accuracy)
```

**Expected Outputs**:
- Training logs with per-round metrics
- Checkpoints of global model
- Convergence information
- Client-specific metrics

---

### Phase 5: Evaluation & Analysis (2-3 hours)
**File**: `notebooks/03_federated_evaluation.ipynb`

**Objectives**:
- Calculate comprehensive FL metrics
- Analyze convergence behavior
- Measure communication costs
- Evaluate privacy trade-offs

**Key Analysis Areas**:

1. **Convergence Analysis**:
```python
# Plot accuracy vs rounds
plt.plot(rounds, accuracies)
plt.xlabel('Communication Round')
plt.ylabel('Global Model Accuracy')
plt.title('Federated Learning Convergence')

# Calculate convergence metrics:
- Rounds to reach 90% of final accuracy
- Final accuracy achieved
- Variance across rounds
```

2. **Per-Client Performance**:
```python
# For each client:
- Local model accuracy (before aggregation)
- Accuracy on global model
- Data heterogeneity metrics
- Training time
```

3. **Communication Analysis**:
```python
# Calculate:
- Total parameters transmitted
- Data per round per client
- Total communication cost
- Bandwidth requirements

# Estimate:
- Network time for transmissions
- Scalability with more clients
```

4. **Privacy Metrics**:
```python
# Document:
- Data privacy guarantees (no raw data shared)
- Model inversion risk (negligible)
- Membership inference difficulty
- Compare to centralized approach
```

5. **Comparison with Version 1**:
```python
# Create comparison table:
Metric              | Centralized (V1) | Federated (V2) | Difference
Accuracy            | X%               | Y%             | ±Z%
Training Time       | A seconds        | B seconds      | ±C%
Data Privacy        | None             | High           | N/A
Communication Cost  | N/A              | C bytes        | N/A
Model Complexity    | K params         | K params       | Same
```

**Expected Outputs**:
- Convergence curves
- Per-round metrics
- Client performance comparison
- Communication cost analysis
- Privacy-utility trade-off documentation

---

### Phase 6: Comparison & Final Analysis (2-3 hours)
**File**: `notebooks/04_comparison_analysis.ipynb`

**Objectives**:
- Compare V1 (centralized) vs V2 (federated)
- Analyze trade-offs
- Document findings
- Provide recommendations

**Comparison Framework**:

1. **Performance Metrics**:
```
Centralized vs Federated:
- Accuracy: V1 vs V2
- F1-Score: V1 vs V2
- Convergence speed: V1 vs V2
- Overfitting (train-test gap): V1 vs V2
```

2. **System Properties**:
```
Communication:
- V1: No communication (all local)
- V2: N clients × num_rounds × model_size

Privacy:
- V1: No privacy (raw data in one place)
- V2: High privacy (data never shared)

Scalability:
- V1: Requires data centralization
- V2: Scales to distributed clients

Model Updates:
- V1: Single training pass
- V2: Multiple rounds, iterative improvement
```

3. **Visualizations**:
```
1. Accuracy comparison bar plot
2. Convergence curves overlay
3. Communication cost breakdown
4. Privacy-utility scatter plot
5. Per-client accuracy comparison
6. Training time comparison
```

4. **Key Findings**:
```
Document:
- Why might V2 have slightly lower accuracy?
  * Non-IID data distribution
  * Limited local training
  * Aggregation effects
  
- Privacy benefits quantified
- Communication costs analyzed
- Practical scenarios for federated approach
- Scenarios where centralized is better
```

**Expected Outputs**:
- Comprehensive comparison report
- Analysis visualizations
- Documented insights
- Recommendations for use cases

---

## Advanced Topics (Optional Extensions)

### Differential Privacy
```python
# Add noise to gradients for stronger privacy
from tensorflow_privacy import dp_optimizers

# Budget: epsilon, delta
# Guarantees theoretical privacy bounds
```

### Secure Aggregation
```python
# Cryptographic protocols
# Server cannot see individual client updates
# More secure but computationally expensive
```

### Model Compression
```python
# Quantization: reduce parameter precision
# Pruning: remove less important parameters
# Reduces communication cost
```

### Personalized Federated Learning
```python
# Learn global + local models
# Adapt to local data distributions
# Better performance on heterogeneous data
```

---

## Code Quality Standards

### Style & Organization
```python
# All code follows PEP 8
# Type hints for all functions
# Docstrings for classes and methods
# Modular, reusable components
# Clear separation of concerns
```

### Example Structure
```python
"""Federated learning client implementation."""

from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
from sklearn.linear_model import LogisticRegression

class DiabetesClient(fl.client.NumPyClient):
    """Federated learning client for diabetes prediction."""
    
    def __init__(
        self,
        model: LogisticRegression,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: str
    ) -> None:
        """Initialize client with local data and model."""
        # Implementation...
```

### Testing Strategy
```
test_data_utils.py:
- Test data distribution
- Verify non-IID generation
- Check data integrity

test_client.py:
- Test parameter get/set
- Test local training
- Test evaluation

test_server.py:
- Test aggregation logic
- Test round management
- Test convergence detection
```

---

## Dependencies

### Core Libraries
```
flwr[simulation]>=1.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
tensorflow>=2.7.0  # Optional, for NN models
```

### Additional
```
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0      # Interactive plots
pytest>=6.2.0
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Execution Guide

### Step 1: Prepare Data
```bash
python scripts/download_dataset.py
python notebooks/01_data_distribution.ipynb
```

### Step 2: Run Federated Training
```bash
# Option A: Single-machine simulation
python scripts/run_federated_training.py --mode simulation

# Option B: Distributed (3 terminals)
python src/server.py &
python src/client.py 0 &
python src/client.py 1 &
python src/client.py 2
```

### Step 3: Analyze Results
```bash
python notebooks/03_federated_evaluation.ipynb
python notebooks/04_comparison_analysis.ipynb
python scripts/visualize_results.py
```

---

## Key Metrics & Success Criteria

### FL-Specific Metrics
- **Convergence**: Reaches stable accuracy within 15-20 rounds
- **Accuracy Loss**: V2 accuracy ≥ V1 accuracy - 5%
- **Communication**: Total data < 500MB for 10-20 rounds
- **Privacy**: 100% data privacy (no raw data shared)

### Code Quality
- Test coverage: >80%
- Type hints: >90%
- Documentation: 100%
- PEP 8 compliance: 100%

---

## Common Challenges & Solutions

### Challenge 1: Non-IID Data Hurts Convergence
**Problem**: With heterogeneous data, accuracy drops
**Solutions**:
- Use FedProx instead of FedAvg
- Increase local epochs
- Implement personalized FL
- Better data distribution strategy

### Challenge 2: Communication Bottleneck
**Problem**: Many rounds needed for convergence
**Solutions**:
- Increase local epochs (fewer rounds)
- Implement model compression
- Use adaptive learning rates
- Monitor bandwidth usage

### Challenge 3: Debugging Distributed Code
**Problem**: Hard to debug when code runs on multiple clients
**Solutions**:
- Test client code locally first
- Use comprehensive logging
- Single-machine simulation before distribution
- Log all parameter changes

### Challenge 4: Model Inconsistency
**Problem**: Clients have different model weights
**Solutions**:
- Synchronization before/after rounds
- Version control for model definitions
- Checkpointing strategy
- Client-side validation

---

## Important Implementation Guidelines

### Claude Code Usage
Ask Claude to:
1. Generate data distribution code with validation
2. Implement Flower client/server classes
3. Create aggregation strategies
4. Generate visualization code
5. Write comprehensive tests
6. Debug FL-specific issues

### Debugging Strategy
1. Start with single-machine simulation
2. Add extensive logging
3. Validate data distribution first
4. Test client code in isolation
5. Incrementally add clients
6. Monitor all metrics carefully

### When to Commit
```
git commit -m "v2: Setup data distribution with validation"
git commit -m "v2: Implement Flower client and server"
git commit -m "v2: Complete federated training pipeline"
git commit -m "v2: Add comprehensive evaluation and analysis"
git commit -m "v2: Create V1 vs V2 comparison report"
```

---

## Timeline Estimate
- **Phase 1 (Data Distribution)**: 1-2 hours
- **Phase 2 (Client Implementation)**: 2-3 hours
- **Phase 3 (Server Implementation)**: 2-3 hours
- **Phase 4 (Training Execution)**: 1-2 hours
- **Phase 5 (Evaluation)**: 2-3 hours
- **Phase 6 (Comparison & Analysis)**: 2-3 hours
- **Total**: 12-16 hours (can be done over 2-3 weeks)

---

## Resources

- **Flower Docs**: https://flower.ai/docs/
- **FL Papers**: arXiv.org (search "federated learning")
- **TensorFlow Federated**: https://www.tensorflow.org/federated
- **Privacy Research**: Differential privacy papers and tutorials

---

## Success Checklist

- [ ] Non-IID data distribution created and validated
- [ ] Flower client implementation complete
- [ ] Flower server implementation complete
- [ ] Federated training runs successfully
- [ ] Convergence behavior documented
- [ ] Per-round metrics tracked
- [ ] V1 vs V2 comparison completed
- [ ] Privacy analysis documented
- [ ] Communication cost calculated
- [ ] All visualizations created
- [ ] Comprehensive README written
- [ ] Code tested and documented
- [ ] GitHub repository updated

---

## Integration with Version 1

This Version 2 project:
- Uses same dataset and target variable
- Reuses preprocessing logic from V1
- Compares against V1 centralized model
- Applies same evaluation metrics
- Can share visualization utilities
- Builds on V1 as foundation

The main difference is **how** models are trained:
- **V1**: All data centralized, single training pass
- **V2**: Data distributed, iterative aggregation

---

## Future Enhancements

1. **Differential Privacy**: Add noise to gradients
2. **Secure Aggregation**: Cryptographic protection
3. **Model Compression**: Reduce communication
4. **Personalization**: Local + global models
5. **Heterogeneous Models**: Different architectures per client
6. **Asynchronous Updates**: Non-blocking aggregation

---

**Note**: This Version 2 project complements Version 1. Both should be in the same GitHub repository with separate directories for clarity. The comparison between versions is a key strength of this portfolio project.
