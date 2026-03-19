# Claude.md - Version 1: Diabetes Prediction - Basic ML Project

## Project Overview
This is a comprehensive machine learning project for predicting diabetes using the Diabetes Health Indicators Dataset from Kaggle. The project showcases fundamental data science skills including exploratory data analysis, data preprocessing, model training, and evaluation.

## Project Goals
- Build a production-ready diabetes prediction model
- Demonstrate comprehensive EDA and data analysis skills
- Compare multiple ML algorithms and select the best performer
- Create professional visualizations and documentation
- Establish a foundation for Version 2 (Federated Learning)

## Technology Stack
- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras
- **Development**: Jupyter Notebook, VS Code
- **Version Control**: Git, GitHub
- **Testing**: Pytest

## Dataset Information
- **Name**: Diabetes Health Indicators Dataset
- **Source**: Kaggle (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Size**: ~254,000 records × 21 features
- **Target**: Diabetes_binary (3-class: 0, 1, 2)
- **Format**: CSV

## Project Structure
```
v1-basic-ml/
├── notebooks/
│   ├── 01_eda_analysis.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Data Cleaning & Preprocessing
│   ├── 03_model_training.ipynb        # Model Development & Training
│   ├── 04_evaluation.ipynb            # Model Evaluation & Comparison
│   └── 05_final_predictions.ipynb     # Generate Final Predictions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Load & basic exploration
│   ├── preprocessor.py                # Data cleaning & feature engineering
│   ├── models.py                      # Model definitions & training
│   ├── evaluation.py                  # Metrics & evaluation functions
│   ├── visualization.py               # Plotting utilities
│   └── config.py                      # Configuration settings
│
├── data/
│   ├── raw/                          # Original Kaggle data
│   │   └── .gitkeep
│   ├── processed/                    # Cleaned & processed data
│   │   └── .gitkeep
│   └── predictions/                  # Model predictions
│       └── .gitkeep
│
├── results/
│   ├── plots/                        # Saved visualizations
│   │   └── .gitkeep
│   ├── models/                       # Trained model files
│   │   └── .gitkeep
│   └── reports/                      # Analysis reports
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   └── test_models.py
│
├── requirements.txt                   # Project dependencies
├── setup.py                          # Package setup
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules
```

## Implementation Workflow

### Phase 1: Exploratory Data Analysis (2-3 hours)
**File**: `notebooks/01_eda_analysis.ipynb`

**Objectives**:
- Load and inspect the dataset
- Display basic statistics (shape, dtypes, missing values)
- Analyze feature distributions
- Identify outliers and anomalies
- Explore target variable distribution
- Create correlation analysis
- Generate visualization plots

**Key Analysis Points**:
- Dataset dimensions and data types
- Missing value percentages
- Statistical summary (mean, median, std, min, max)
- Class balance in target variable
- Feature distributions (histograms)
- Correlation matrix heatmap
- Outlier detection (box plots)
- Feature-target relationships

**Expected Outputs**:
- Summary statistics DataFrame
- Multiple visualization plots
- Identified data quality issues
- Insights document

---

### Phase 2: Data Preprocessing (2-3 hours)
**File**: `notebooks/02_preprocessing.ipynb`

**Objectives**:
- Handle missing values
- Detect and treat outliers
- Feature scaling and normalization
- Categorical variable encoding (if any)
- Address class imbalance
- Feature engineering
- Train-test split

**Key Tasks**:
1. **Missing Values**:
   - Check percentages
   - Implement mean imputation or removal
   - Document decisions

2. **Outliers**:
   - Use IQR method or Z-score
   - Decide on treatment (removal or capping)
   - Log transformations if needed

3. **Feature Scaling**:
   - Apply StandardScaler for ML models
   - Apply MinMaxScaler for neural networks (optional)
   - Fit on training, apply to test

4. **Encoding**:
   - Label encode if categorical features exist
   - One-hot encode if necessary
   - Document encoding mappings

5. **Class Imbalance**:
   - Check class distribution
   - Consider SMOTE oversampling
   - Use class weights in model training
   - Stratified split for train-test

6. **Train-Test Split**:
   - 80-20 or 70-30 split
   - Use stratification
   - Save split indices for reproducibility

**Expected Outputs**:
- Preprocessed train/test datasets
- Scaler fitted and saved
- Preprocessing pipeline documented
- Validation checks performed

---

### Phase 3: Model Training & Development (3-4 hours)
**File**: `notebooks/03_model_training.ipynb`

**Objectives**:
- Train multiple baseline models
- Implement hyperparameter tuning
- Use cross-validation
- Track training metrics
- Save trained models

**Models to Implement**:

1. **Logistic Regression** (Baseline)
   - Quick to train
   - Interpretable coefficients
   - Use for comparison

2. **Random Forest**
   - Feature importance analysis
   - Handles non-linearities
   - Default hyperparameters first

3. **XGBoost or LightGBM**
   - Best gradient boosting performance
   - Feature importance from trees
   - Hyperparameter tuning critical

4. **Neural Network** (Keras/TensorFlow)
   - Demonstrate deep learning
   - 2-3 hidden layers
   - Early stopping and validation

**Training Strategy**:
```python
# For each model:
1. Initialize with default parameters
2. Train on training set
3. Evaluate on validation set using cross-validation
4. Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
5. Retrain with best parameters
6. Evaluate on test set
7. Save trained model
8. Log training metrics
```

**Hyperparameter Tuning**:
- Use GridSearchCV or RandomizedSearchCV
- 5-fold cross-validation
- Save best parameters
- Document tuning results

**Expected Outputs**:
- Trained model objects (pickle files)
- Cross-validation scores
- Best hyperparameters for each model
- Training logs with metrics
- Comparison table

---

### Phase 4: Model Evaluation & Comparison (1-2 hours)
**File**: `notebooks/04_evaluation.ipynb`

**Objectives**:
- Calculate comprehensive metrics
- Generate evaluation plots
- Compare all models
- Select best model
- Analyze model performance

**Metrics to Calculate**:

**Classification Metrics**:
- Accuracy
- Precision (macro, weighted)
- Recall (macro, weighted)
- F1-Score (macro, weighted)
- Specificity
- ROC-AUC
- PR-AUC

**For Each Class**:
- Per-class precision, recall, F1
- Confusion matrix
- True positives, false positives, etc.

**Additional Analysis**:
- Learning curves (training size vs performance)
- Feature importance ranking
- Cross-validation stability
- Training vs test performance gap

**Visualizations to Create**:
```
1. Confusion matrices (heatmap for each model)
2. ROC curves (overlay all models)
3. Precision-Recall curves
4. Feature importance plots (top 15 features)
5. Model comparison bar plots:
   - Accuracy comparison
   - F1-Score comparison
   - ROC-AUC comparison
6. Learning curves
7. Classification reports (per class)
```

**Model Selection Criteria**:
- Primary: ROC-AUC or F1-Score (based on business need)
- Secondary: Training time, model interpretability
- Check for overfitting (train vs test gap)
- Consider class-specific performance

**Expected Outputs**:
- Comprehensive metrics DataFrame
- Evaluation plots (PNG files)
- Model comparison report
- Selected best model specification
- Performance documentation

---

### Phase 5: Final Predictions & Documentation (1-2 hours)
**File**: `notebooks/05_final_predictions.ipynb`

**Objectives**:
- Generate predictions on test set
- Create predictions file with confidence scores
- Write final analysis report
- Document all findings

**Prediction Generation**:
```python
# For best model:
1. Load trained model
2. Generate predictions (class labels)
3. Generate prediction probabilities
4. Create DataFrame with:
   - Original features
   - Predicted class
   - Probability for each class
   - Confidence score (max probability)
5. Export to CSV
```

**Analysis Report Should Include**:
- Executive summary
- Dataset overview
- Data preprocessing decisions
- Model selection methodology
- Final model performance
- Key insights and findings
- Limitations and assumptions
- Future improvements

**Expected Outputs**:
- predictions.csv with confidence scores
- Final analysis report (markdown or PDF)
- Model card/documentation
- README update with results

---

## Code Quality Standards

### Style & Format
- Follow PEP 8 style guide
- Use type hints for functions
- Docstrings for all functions and classes
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable names

### Project Structure Best Practices
```python
# Example: data_loader.py
"""Module for loading and basic data exploration."""

import pandas as pd
import numpy as np
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load diabetes dataset from CSV.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")

def basic_info(df: pd.DataFrame) -> dict:
    """Get basic information about dataset."""
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'duplicates': df.duplicated().sum()
    }
```

### Testing
- Create unit tests for preprocessing functions
- Test edge cases and error handling
- Mock data for tests
- Run tests before committing

---

## Key Metrics & Success Criteria

### Model Performance Targets
- **Baseline Accuracy**: >70%
- **Target ROC-AUC**: >0.75
- **Target F1-Score**: >0.65 (weighted)
- **Cross-Val Stability**: Std Dev < 0.05

### Code Quality Metrics
- Test coverage: >80%
- Type hint coverage: >90%
- Docstring coverage: 100%
- PEP 8 compliance: 100%

### Documentation
- Comprehensive README (>500 words)
- Code examples in README
- Setup instructions clear
- Results documented with figures

---

## Important Implementation Guidelines

### Claude Code Usage
1. **Ask Claude to generate code** for:
   - Data loading and exploration
   - Preprocessing pipelines
   - Model training loops
   - Evaluation metrics calculation

2. **Ask Claude to create visualizations** using Matplotlib/Seaborn:
   - Correlation heatmaps
   - Feature distributions
   - ROC curves
   - Confusion matrices

3. **Ask Claude to structure your project**:
   - Create modular functions
   - Write docstrings
   - Suggest improvements

### Workflow Recommendations
1. Start with one Jupyter notebook per phase
2. Convert successful code to reusable modules
3. Use Claude to generate test cases
4. Ask Claude for code reviews
5. Let Claude help with error debugging

### When to Commit to Git
- After completing each phase
- After successful test runs
- Before major refactoring
- With descriptive commit messages

**Example commits**:
```
git commit -m "v1: Complete EDA with correlation and distribution analysis"
git commit -m "v1: Implement preprocessing pipeline with scaling and encoding"
git commit -m "v1: Add RandomForest and XGBoost model training"
git commit -m "v1: Complete model evaluation with visualization"
git commit -m "v1: Generate final predictions and analysis report"
```

---

## Common Challenges & Solutions

### Challenge 1: Class Imbalance
**Problem**: Target variable is imbalanced (unequal class distribution)
**Solutions**:
- Use SMOTE oversampling in preprocessing
- Use class_weight='balanced' in model training
- Use stratified cross-validation
- Choose evaluation metric (F1 over accuracy)

### Challenge 2: Feature Scaling
**Problem**: Some features have vastly different scales
**Solution**:
- Use StandardScaler from sklearn
- Fit on training data only
- Apply to test data
- Consider separate scaling for different model types

### Challenge 3: Model Overfitting
**Problem**: Training accuracy >> test accuracy
**Solutions**:
- Use cross-validation to detect early
- Add regularization (L1/L2)
- Reduce model complexity
- Use early stopping for neural networks
- Create learning curves to visualize

### Challenge 4: Reproducibility
**Problem**: Results differ between runs
**Solutions**:
- Set random seeds: `np.random.seed(42)`, `tf.random.set_seed(42)`
- Document all preprocessing steps
- Save feature names and scaler
- Include requirements.txt with versions

---

## Dependencies

### Core Libraries
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
jupyter-notebook>=6.0.0
```

### Machine Learning
```
xgboost>=1.5.0
lightgbm>=3.3.0
tensorflow>=2.7.0
```

### Development
```
black>=21.0
pylint>=2.10.0
pytest>=6.2.0
pytest-cov>=2.12.0
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Next Steps After Completion

1. **Create GitHub repository**
   - Add comprehensive README
   - Include all notebooks and source code
   - Add results and visualizations

2. **Write Blog Post**
   - Explain the problem
   - Document approach
   - Share key findings
   - Include visualizations

3. **Prepare for Version 2**
   - Understand federated learning basics
   - Plan federated architecture
   - Prepare data distribution strategy

4. **Interview Preparation**
   - Prepare to discuss decisions
   - Explain algorithm choices
   - Talk about trade-offs
   - Discuss improvements

---

## Resources

- **Pandas Docs**: https://pandas.pydata.org/docs/
- **Scikit-learn Guide**: https://scikit-learn.org/
- **XGBoost Tutorial**: https://xgboost.readthedocs.io/
- **TensorFlow/Keras**: https://tensorflow.org/guide
- **Matplotlib Guide**: https://matplotlib.org/
- **Seaborn Gallery**: https://seaborn.pydata.org/

---

## Timeline Estimate
- **Phase 1 (EDA)**: 2-3 hours
- **Phase 2 (Preprocessing)**: 2-3 hours
- **Phase 3 (Model Training)**: 3-4 hours
- **Phase 4 (Evaluation)**: 1-2 hours
- **Phase 5 (Final & Docs)**: 1-2 hours
- **Total**: 10-14 hours (can be done over 1-2 weeks)

---

## Claude Code Prompt Examples

### For EDA
```
"Generate Python code to load the diabetes dataset and create:
1. Summary statistics
2. Missing value analysis
3. Distribution plots for all features
4. Correlation heatmap
5. Box plots for outlier detection"
```

### For Preprocessing
```
"Create a preprocessing pipeline that:
1. Handles missing values using mean imputation
2. Detects and removes outliers using IQR method
3. Scales features using StandardScaler
4. Addresses class imbalance using SMOTE
5. Splits data into train/test with stratification"
```

### For Model Training
```
"Train multiple models (Logistic Regression, Random Forest, XGBoost):
1. Use 5-fold cross-validation
2. Perform hyperparameter tuning with GridSearchCV
3. Track training and validation metrics
4. Save trained models
5. Create a comparison table"
```

### For Evaluation
```
"Create evaluation functions that calculate:
1. Accuracy, Precision, Recall, F1-score
2. ROC-AUC and confusion matrix
3. Per-class metrics
4. Plot ROC curves for all models
5. Compare models visually"
```

---

## Success Checklist

- [ ] Dataset downloaded and loaded successfully
- [ ] EDA notebook completed with all visualizations
- [ ] Preprocessing pipeline implemented and documented
- [ ] At least 3 models trained and compared
- [ ] Hyperparameter tuning performed
- [ ] Test metrics calculated and documented
- [ ] Visualizations created (heatmaps, ROC curves, etc.)
- [ ] Code is modular and well-documented
- [ ] Tests written and passing
- [ ] README updated with results
- [ ] GitHub repository created and pushed
- [ ] All notebooks run end-to-end without errors

---

**Note**: This Version 1 project establishes the foundation for Version 2 (Federated Learning). The models, preprocessing, and evaluation strategies developed here will be adapted for the federated learning scenario.
