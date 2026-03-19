# ML Portfolio Projects - Quick Reference Guide

## 🎯 Project Overview
- **Dataset**: Diabetes Health Indicators (Kaggle)
- **Version 1**: Basic ML/Data Science Project (40-50 hours)
- **Version 2**: Federated Learning Extension (40-50 hours)
- **Total Timeline**: 3-4 weeks, 80-90 hours

---

## 📊 Dataset Information

**Name**: Diabetes Health Indicators Dataset
**Link**: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
**Size**: ~254,000 records × 21 features
**Target**: 3-class classification (No diabetes, Prediabetes, Diabetes)

### Why This Dataset?
✓ Perfect size for demonstration projects
✓ Real-world healthcare data with clear business value
✓ Multiple ML algorithm opportunities
✓ Ideal for federated learning scenarios
✓ Strong portfolio appeal

---

## ⚙️ Setup Instructions

### Environment Setup
```bash
# Create virtual environment
python -m venv diabetes_env
source diabetes_env/bin/activate  # Windows: diabetes_env\Scripts\activate

# Install Version 1 dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter kaggle tensorflow

# Install Version 2 dependencies (in addition to above)
pip install flwr[simulation]

# Configure Kaggle
# 1. Download kaggle.json from https://www.kaggle.com/settings/account
# 2. Place in ~/.kaggle/kaggle.json
# 3. Run: chmod 600 ~/.kaggle/kaggle.json  # On Linux/Mac
```

### Download Dataset
```bash
pip install kaggle
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset
unzip diabetes-health-indicators-dataset.zip
```

---

## 📝 Version 1: Basic ML Project

### Workflow
1. **EDA** (2-3 hours) → Jupyter Notebook
2. **Preprocessing** (2-3 hours) → Python script
3. **Model Training** (3-4 hours) → Multiple algorithms
4. **Evaluation** (1-2 hours) → Metrics & visualizations
5. **Documentation** (2-3 hours) → README & analysis

### Key Deliverables
- Jupyter notebooks with step-by-step analysis
- Trained model files (.pkl, .h5)
- Prediction CSV with confidence scores
- Visualizations (correlation matrix, ROC curves, etc.)
- Comprehensive README with findings

### Models to Implement
```
1. Logistic Regression (baseline)
2. Random Forest (feature importance)
3. XGBoost or LightGBM (gradient boosting)
4. Neural Network (TensorFlow/Keras)
```

### Code Template: Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Data preparation
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
test_score = model.score(X_test_scaled, y_test)
print(f'CV Mean: {cv_scores.mean():.4f}, Test: {test_score:.4f}')
```

### Evaluation Metrics Checklist
- [ ] Accuracy
- [ ] Precision, Recall, F1-score (per class)
- [ ] Confusion Matrix
- [ ] ROC-AUC and ROC Curve
- [ ] Precision-Recall Curve
- [ ] Cross-validation scores
- [ ] Feature importance

---

## 🔗 Version 2: Federated Learning

### Workflow
1. **Data Distribution** (1-2 hours) → Split into non-IID chunks
2. **Framework Setup** (1-2 hours) → Install Flower
3. **Client Implementation** (2-3 hours) → Local training logic
4. **Server Implementation** (2-3 hours) → Aggregation logic
5. **Training & Analysis** (3-4 hours) → Run FL and analyze
6. **Comparison** (2-3 hours) → V1 vs V2 comparison

### Non-IID Data Distribution Setup
```python
import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')

# Dirichlet distribution for realistic non-IID
n_clients = 3
alpha = 0.1  # Lower = more heterogeneous

np.random.seed(42)
client_proportions = np.random.dirichlet([alpha] * n_clients)

# Distribute data
for client_id in range(n_clients):
    n_samples = int(client_proportions[client_id] * len(df))
    indices = np.random.choice(len(df), n_samples, replace=False)
    client_df = df.iloc[indices]
    client_df.to_csv(f'client_{client_id}_data.csv', index=False)
    print(f'Client {client_id}: {len(client_df)} samples, '
          f'{client_df["Diabetes_binary"].value_counts().to_dict()}')
```

### Flower Client Implementation Template
```python
import flwr as fl
from sklearn.linear_model import LogisticRegression
import numpy as np

class DiabetesClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        """Return model parameters."""
        return [self.model.coef_, self.model.intercept_]

    def fit(self, parameters, config):
        """Train model locally."""
        self.model.coef_, self.model.intercept_ = parameters
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test set."""
        self.model.coef_, self.model.intercept_ = parameters
        loss = 1 - self.model.score(self.X_test, self.y_test)
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

# Start client
if __name__ == "__main__":
    client = DiabetesClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
```

### Flower Server Implementation Template
```python
import flwr as fl
from flwr.server.strategy import FedAvg
from sklearn.linear_model import LogisticRegression

# Initialize global model
initial_model = LogisticRegression(max_iter=1000)
initial_params = [initial_model.coef_, initial_model.intercept_]

# Setup strategy with weighted aggregation
strategy = FedAvg(
    fraction_fit=1.0,  # Use all available clients
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    initial_parameters=initial_params,
)

# Start server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10),
)
```

### Running Federated Learning
```bash
# Terminal 1: Start server
python server.py

# Terminal 2, 3, 4: Start clients
python client.py 0  # Client 0
python client.py 1  # Client 1
python client.py 2  # Client 2

# Wait for convergence...
```

### Key FL Concepts
- **FederatedAveraging**: Weight average based on data size
- **Non-IID Data**: Heterogeneous distributions across clients
- **Communication Rounds**: Number of iterations
- **Local Epochs**: Training steps per client per round
- **Privacy Budget**: Differential privacy consideration

---

## 📁 Repository Structure

```
diabetes-ml-portfolio/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/diabetes.csv (from Kaggle)
│   ├── processed/train_processed.csv
│   └── federated/client_0_data.csv, etc.
│
├── v1-basic-ml/
│   ├── notebooks/
│   │   ├── 01_eda_analysis.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_evaluation.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── models.py
│   │   └── evaluation.py
│   └── results/
│       ├── correlation_matrix.png
│       ├── roc_curves.png
│       └── model_comparison.png
│
├── v2-federated-learning/
│   ├── notebooks/
│   │   ├── 01_data_distribution.ipynb
│   │   └── 02_federated_training.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── data_utils.py
│   │   └── models.py
│   └── results/
│       ├── convergence_plot.png
│       └── fl_vs_centralized.png
│
├── docs/
│   ├── setup_guide.md
│   ├── dataset_description.md
│   ├── implementation_notes.md
│   └── results_summary.md
│
└── tests/
    ├── test_preprocessing.py
    └── test_models.py
```

---

## 📊 Key Metrics to Track

### Version 1 (Baseline)
```
Accuracy: [result]
Precision: [per class]
Recall: [per class]
F1-Score: [per class]
AUC-ROC: [score]
Cross-Val Mean: [score ± std]
Training Time: [X seconds]
Inference Time: [X ms]
```

### Version 2 (Federated)
```
Global Model Accuracy by Round: [plot]
Client Local Accuracy: [comparison]
Communication Rounds to Convergence: [number]
Communication Cost: [bytes transferred]
Privacy-Utility Trade-off: [analysis]
Training Time (FL vs Centralized): [comparison]
Accuracy Difference: [V1 - V2]
```

---

## 🚀 GitHub Best Practices

### .gitignore Template
```
# Environment
venv/
env/
diabetes_env/

# Data
data/raw/
*.csv
*.pkl
*.h5

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### Commit Message Examples
```
git add .
git commit -m "v1: Complete EDA and visualization analysis"
git commit -m "v1: Implement and tune RandomForest model"
git commit -m "v2: Setup Flower framework and client code"
git commit -m "v2: Implement federated server aggregation"
git commit -m "docs: Add comprehensive README and results analysis"
```

### README Template Headers
```markdown
# Diabetes Prediction - ML Portfolio Project

## Overview
[Project description]

## Quick Start
[Installation and usage]

## Version 1: Basic ML
[EDA, models, results]

## Version 2: Federated Learning
[FL approach, implementation, results]

## Results
[Key findings and comparisons]

## Repository Structure
[File organization]

## How to Reproduce
[Step-by-step instructions]

## Future Improvements
[V3+ ideas]

## References
[Papers and resources]
```

---

## 📈 Visualization Checklist

### Version 1 Plots
- [ ] Feature distributions (histograms)
- [ ] Correlation heatmap
- [ ] Missing values heatmap
- [ ] Class distribution (pie chart)
- [ ] Box plots for outliers
- [ ] Feature importance (bar plot)
- [ ] Confusion matrix heatmap
- [ ] ROC curves (all models)
- [ ] Precision-Recall curves
- [ ] Learning curves
- [ ] Model comparison bar plot
- [ ] Training time comparison

### Version 2 Additional Plots
- [ ] Data distribution per client (stacked bar)
- [ ] Convergence curves (accuracy vs rounds)
- [ ] FL vs Centralized comparison
- [ ] Communication cost analysis
- [ ] Privacy-utility trade-off curve
- [ ] Local vs global model accuracy
- [ ] Training time comparison

---

## 🎓 Interview Talking Points

**On Dataset Choice:**
- "I chose healthcare data because it's relevant, has clear business value, and naturally motivates federated learning."

**On Version 1:**
- "I performed comprehensive EDA to understand the data, handled class imbalance using [method], and compared multiple algorithms to find the best performer."

**On Version 2:**
- "I extended the project to federated learning to demonstrate privacy-preserving ML, a critical concern in modern data science."

**On Challenges:**
- "One challenge was [specific issue], which I solved by [solution]. This taught me [lesson]."

**On Trade-offs:**
- "Federated learning trades some accuracy for significant privacy benefits. The [X]% accuracy difference is acceptable given the privacy guarantees."

---

## 📚 Essential Resources

- **Kaggle Dataset**: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **Flower Documentation**: https://flower.ai/docs/
- **scikit-learn Guide**: https://scikit-learn.org/
- **TensorFlow/Keras**: https://tensorflow.org/guide
- **Pandas Documentation**: https://pandas.pydata.org/docs/

---

## 📋 Pre-Submission Checklist

### Code Quality
- [ ] All code is PEP 8 compliant
- [ ] Functions have docstrings
- [ ] Variables have meaningful names
- [ ] Code is modular and reusable
- [ ] Error handling implemented

### Documentation
- [ ] Comprehensive README written
- [ ] Installation instructions clear
- [ ] Results documented with figures
- [ ] Code comments explain logic
- [ ] References cited

### Repository
- [ ] .gitignore properly configured
- [ ] requirements.txt up to date
- [ ] README has badges (optional but nice)
- [ ] All notebooks run end-to-end
- [ ] No uncommitted changes

### Analysis
- [ ] EDA insights documented
- [ ] Model comparisons completed
- [ ] V1 vs V2 analysis done
- [ ] Future improvements listed
- [ ] Limitations discussed

---

**Good luck with your portfolio project! This will be an excellent addition to your professional profile.**
