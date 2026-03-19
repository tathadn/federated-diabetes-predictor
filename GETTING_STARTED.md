# 🚀 Getting Started: Complete Setup Guide

## Overview
You now have a complete portfolio project package for building two versions of a Diabetes Prediction ML system:
- **Version 1**: Traditional centralized ML approach
- **Version 2**: Federated Learning (distributed, privacy-preserving)

This guide walks you through setting everything up and starting your development.

---

## 📦 Files You Have

### Core Project Files
1. **ML_Portfolio_Projects_Guide.pdf** (15 pages)
   - Comprehensive project overview
   - Dataset recommendations
   - Implementation steps for both versions
   - Code examples and architecture diagrams
   
2. **QUICK_REFERENCE.md**
   - Quick setup commands
   - Code snippets
   - Checklists
   - Quick lookup for common tasks

### Claude Code Files (For VS Code Development)
3. **V1_CLAUDE.md**
   - Complete specification for Version 1
   - Project goals and structure
   - 5 implementation phases
   - Code quality standards
   - Success criteria
   
4. **V2_CLAUDE.md**
   - Complete specification for Version 2
   - Federated learning concepts
   - 6 implementation phases
   - Advanced topics
   - Debugging strategies

5. **CLAUDE_CODE_GUIDE.md**
   - How to use Claude Code in VS Code
   - Installation instructions
   - Workflow examples
   - Best practices
   - Keyboard shortcuts

6. **CLAUDE_CODE_EXAMPLES.md**
   - Practical example prompts for Claude
   - Expected outputs
   - Copy-paste ready prompts
   - Debugging examples
   - Optimization techniques

---

## 📋 Step-by-Step Setup

### Step 1: Set Up Your Local Environment (15 minutes)

```bash
# Create project directory
mkdir diabetes-ml-portfolio
cd diabetes-ml-portfolio

# Initialize Git
git init
git config user.name "Your Name"
git config user.email "your@email.com"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter kaggle
pip install xgboost lightgbm tensorflow
pip install flwr[simulation]  # For Version 2
pip install pytest black pylint  # For development
```

### Step 2: Configure Kaggle API (10 minutes)

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. This downloads kaggle.json

# On Windows (PowerShell):
mkdir $env:USERPROFILE\.kaggle
Move-Item .\kaggle.json $env:USERPROFILE\.kaggle\

# On Mac/Linux:
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify it works
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset
unzip diabetes-health-indicators-dataset.zip
```

### Step 3: Create Project Structure (10 minutes)

```bash
# Create directories for Version 1
mkdir -p v1-basic-ml/{notebooks,src,data/raw,data/processed,results/plots,results/models,results/reports,tests}

# Create directories for Version 2
mkdir -p v2-federated-learning/{notebooks,src,data/federated,results/plots,results/models,results/logs,tests,scripts,configs}

# Create shared directories
mkdir -p docs

# Create .gitignore files
echo ".gitkeep" > v1-basic-ml/data/raw/.gitkeep
echo ".gitkeep" > v2-federated-learning/data/federated/.gitkeep

# List structure
tree -L 3  # or 'ls -R' on Windows
```

### Step 4: Install VS Code & Claude Code Extension (15 minutes)

**If you don't have VS Code:**
1. Download from https://code.visualstudio.com/
2. Install it

**Install Claude Code Extension:**
1. Open VS Code
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
3. Search for "Claude"
4. Click "Install" on the official Anthropic extension
5. Sign in with your Claude account

### Step 5: Copy Claude.md Files to Project

```bash
# Copy Version 1 claude.md
cp V1_CLAUDE.md v1-basic-ml/claude.md

# Copy Version 2 claude.md  
cp V2_CLAUDE.md v2-federated-learning/claude.md

# Verify files are in place
ls v1-basic-ml/claude.md
ls v2-federated-learning/claude.md
```

### Step 6: Create Initial Files

```bash
# Create requirements.txt for V1
cat > requirements_v1.txt << EOF
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
tensorflow>=2.7.0
pytest>=6.2.0
black>=21.0
pylint>=2.10.0
kaggle>=1.5.0
EOF

# Create requirements.txt for V2 (includes V1)
cat > requirements_v2.txt << EOF
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
jupyter>=1.0.0
tensorflow>=2.7.0
flwr[simulation]>=1.0.0
pytest>=6.2.0
black>=21.0
kaggle>=1.5.0
EOF

# Create .gitignore
cat > .gitignore << EOF
# Virtual Environment
venv/
env/
*.egg-info/

# Data files
data/raw/
data/federated/
*.csv
*.xlsx

# Models
*.pkl
*.h5
*.pth

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*.so
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/
EOF

git add .gitignore
```

### Step 7: Initial Git Commit

```bash
# Create README (we'll expand this later)
cat > README.md << EOF
# Diabetes ML Portfolio Project

This repository contains two versions of a diabetes prediction machine learning project:
- **Version 1**: Traditional centralized ML approach
- **Version 2**: Federated Learning (privacy-preserving)

See individual project directories for details.
EOF

# Initial commit
git add README.md v1-basic-ml/claude.md v2-federated-learning/claude.md requirements_v1.txt requirements_v2.txt
git commit -m "Initial project setup with claude.md specifications"

# Verify git status
git log --oneline
```

---

## 🎯 Starting Your Development

### Opening Version 1 in VS Code

```bash
# From diabetes-ml-portfolio directory
code v1-basic-ml/

# This opens VS Code with V1 folder
# Claude Code automatically reads claude.md
```

**In VS Code:**
1. Open Claude Code panel (Ctrl+Shift+L)
2. Claude has read your V1_CLAUDE.md
3. Ready to generate code!

### Opening Version 2 in VS Code

```bash
code v2-federated-learning/

# Now Claude Code reads V2_CLAUDE.md
```

---

## 💻 Development Workflow

### For Version 1 (Week 1-2)

**Day 1-2: EDA Phase**
```bash
# Open VS Code with v1-basic-ml folder
code v1-basic-ml/

# In Claude Code chat (Ctrl+Shift+L), ask:
"Generate 01_eda_analysis.ipynb according to my claude.md Phase 1.
Include all sections and visualizations."

# Copy generated notebook to notebooks/01_eda_analysis.ipynb
# Run the notebook in Jupyter
```

**Day 3-4: Preprocessing**
```bash
# Ask Claude in V1 folder:
"Create src/preprocessor.py according to Phase 2 of my claude.md.
Include class structure, methods, and docstrings."

# Copy to src/preprocessor.py
# Ask for tests:
"Generate tests/test_preprocessor.py with >80% coverage"

# Run tests: pytest tests/test_preprocessor.py
```

**Day 5-7: Model Training**
```bash
# Ask Claude:
"Create src/models.py with all models from Phase 3.
Include training loop, cross-validation, and hyperparameter tuning."

# Copy and create notebook:
"Generate 03_model_training.ipynb using src/models.py"

# Run training
```

**Day 8-10: Evaluation & Documentation**
```bash
# Ask Claude:
"Create 04_evaluation.ipynb for Phase 4 evaluation.
Include all metrics and comparison visualizations."

"Create comprehensive README.md for Version 1"

# Run evaluation and finalize
```

### For Version 2 (Week 3-4)

**Day 1-2: Data Distribution**
```bash
code v2-federated-learning/

# Ask Claude:
"Create src/data_utils.py according to Phase 1.
Implement non-IID data distribution with Dirichlet sampling."

# Test with notebook:
"Create 01_data_distribution.ipynb to visualize client data"
```

**Day 3-4: Federated Implementation**
```bash
# Ask Claude:
"Generate src/client.py implementing Flower federated client (Phase 2)"

"Generate src/server.py with FedAvg strategy (Phase 3)"

"Create scripts/run_federated_training.py to execute FL (Phase 4)"
```

**Day 5-6: Training & Analysis**
```bash
# Run federated training:
python scripts/run_federated_training.py --num_rounds 15

# Ask Claude for analysis:
"Create 03_federated_evaluation.ipynb for Phase 5 analysis"

"Create 04_comparison_analysis.ipynb comparing V1 vs V2 (Phase 6)"
```

**Day 7-10: Finalization**
```bash
# Ask Claude:
"Create comprehensive README.md for Version 2"

"Create comparison document: V1_vs_V2_Analysis.md"

# Final cleanup and polish
```

---

## 📚 Using the Documentation

### When You Need Help
1. **Quick answer?** → QUICK_REFERENCE.md
2. **Understanding concept?** → PDF guide
3. **Implementation details?** → Relevant claude.md file
4. **Code examples?** → CLAUDE_CODE_EXAMPLES.md
5. **How to use Claude Code?** → CLAUDE_CODE_GUIDE.md

### Workflow: Reading Files

**Starting V1:**
```
1. Read "Overview" section in V1_CLAUDE.md
2. Skim "Project Structure" to understand layout
3. Read Phase 1 in detail
4. Reference CLAUDE_CODE_EXAMPLES.md for example prompts
5. Ask Claude to generate Phase 1 code
```

**Starting V2:**
```
1. Read "Key Federated Learning Concepts" in V2_CLAUDE.md
2. Understand "Project Structure"
3. Read Phase 1 (data distribution) carefully
4. Follow same pattern for each subsequent phase
5. Reference CLAUDE_CODE_GUIDE.md for debugging tips
```

---

## 🛠️ Common Commands

### Git Workflow
```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "v1: [phase] - description"

# View history
git log --oneline

# Push to GitHub (after setup)
git remote add origin https://github.com/yourusername/diabetes-ml-portfolio.git
git push -u origin main
```

### Running Code
```bash
# Install dependencies
pip install -r requirements_v1.txt

# Run Jupyter
jupyter notebook

# Run Python script
python src/data_loader.py

# Run tests
pytest tests/ -v --cov=src

# Run VS Code in project
code .
```

---

## 📊 Tracking Progress

### Version 1 Checklist

**Phase 1: EDA**
- [ ] Data loaded successfully
- [ ] Basic statistics computed
- [ ] Distributions visualized
- [ ] Correlation analysis done
- [ ] Outliers identified
- [ ] Notebook completed and saved
- [ ] Commit: "v1: Complete EDA"

**Phase 2: Preprocessing**
- [ ] Missing values handled
- [ ] Outliers treated
- [ ] Features scaled
- [ ] Class imbalance addressed
- [ ] Train-test split created
- [ ] Module created and tested
- [ ] Commit: "v1: Complete preprocessing"

**Phase 3: Model Training**
- [ ] Baseline model trained
- [ ] RF, XGBoost, NN implemented
- [ ] Cross-validation working
- [ ] Hyperparameter tuning done
- [ ] All models saved
- [ ] Notebook completed
- [ ] Commit: "v1: Complete model training"

**Phase 4: Evaluation**
- [ ] All metrics calculated
- [ ] Confusion matrices created
- [ ] ROC curves plotted
- [ ] Feature importance shown
- [ ] Models compared
- [ ] Best model selected
- [ ] Commit: "v1: Complete evaluation"

**Phase 5: Documentation**
- [ ] Predictions generated
- [ ] Analysis report written
- [ ] README completed
- [ ] All code documented
- [ ] Tests passing
- [ ] Final commit
- [ ] Ready for GitHub

### Version 2 Checklist

**Phase 1: Data Distribution**
- [ ] Non-IID distribution created
- [ ] Client datasets validated
- [ ] Statistics documented
- [ ] Distribution visualized
- [ ] Commit: "v2: Complete data distribution"

**Phase 2: Client Implementation**
- [ ] Client class implemented
- [ ] Local training working
- [ ] Parameter update logic correct
- [ ] Logging comprehensive
- [ ] Tests passing
- [ ] Commit: "v2: Complete client"

**Phase 3: Server Implementation**
- [ ] Server initialized
- [ ] FedAvg strategy configured
- [ ] Aggregation logic correct
- [ ] Checkpointing working
- [ ] Tests passing
- [ ] Commit: "v2: Complete server"

**Phase 4: FL Training**
- [ ] Training script functional
- [ ] Simulation mode working
- [ ] Metrics tracked
- [ ] Models saved
- [ ] Convergence achieved
- [ ] Commit: "v2: Complete FL training"

**Phase 5: Evaluation**
- [ ] Convergence analyzed
- [ ] Per-client performance evaluated
- [ ] Communication cost calculated
- [ ] Privacy benefits documented
- [ ] Visualizations created
- [ ] Commit: "v2: Complete evaluation"

**Phase 6: Comparison**
- [ ] V1 vs V2 analysis done
- [ ] Comparison report written
- [ ] Trade-offs documented
- [ ] Recommendations provided
- [ ] README completed
- [ ] Final commit
- [ ] Ready for GitHub

---

## 🚀 Pushing to GitHub

### When You're Ready

```bash
# Create repo on GitHub first (without initialization)
# Then:

git remote add origin https://github.com/yourusername/diabetes-ml-portfolio.git
git branch -M main
git push -u origin main

# Verify on GitHub
# Should see all your commits and files
```

### GitHub Profile Enhancement

- ⭐ Pin the repository
- 📝 Add repository description
- 🏷️ Add topics: machine-learning, data-science, federated-learning
- 📄 Ensure README is comprehensive
- 🔗 Add to portfolio website

---

## 📖 Recommended Reading Order

### Before Starting
1. PDF Guide - Overview section (understand the big picture)
2. CLAUDE_CODE_GUIDE.md - How Claude Code works
3. V1_CLAUDE.md - Read overview, project goals, workflow

### During V1 Development
- Reference V1_CLAUDE.md for each phase
- Use CLAUDE_CODE_EXAMPLES.md for prompt ideas
- Check QUICK_REFERENCE.md for syntax help

### Before Starting V2
1. V2_CLAUDE.md - Read key concepts section carefully
2. Research federated learning basics (optional)
3. Understand non-IID data distribution

### During V2 Development
- Deep dive into V2_CLAUDE.md phases
- Use CLAUDE_CODE_EXAMPLES.md Part 2 (V2 examples)
- Reference CLAUDE_CODE_GUIDE.md for debugging

---

## 💡 Pro Tips

### Tip 1: Backup Your Work
```bash
# Push to GitHub frequently
git push

# Don't rely on local machine only
```

### Tip 2: Test as You Go
```bash
# After each phase, run tests
pytest tests/ -v

# Before committing, verify nothing is broken
```

### Tip 3: Document Discoveries
```bash
# Add notes as you go
# Use markdown cells in notebooks
# Update README as you progress
```

### Tip 4: Ask Claude Strategic Questions
```
Don't just ask for code. Ask:
- "Why is this approach better?"
- "What are the trade-offs?"
- "How does this relate to my requirements?"
```

### Tip 5: Review Generated Code
```
Never blindly accept Claude's code:
- Read it carefully
- Understand what it does
- Test it thoroughly
- Make adjustments as needed
```

---

## 🎓 Learning Path

If new to these topics:

**Before Starting:**
- Python basics (if needed)
- Pandas/NumPy tutorials (1-2 hours)
- Scikit-learn intro (1-2 hours)

**Version 1:**
- Teaches: ML fundamentals, EDA, model training
- Expected: 10-14 hours of active coding

**Version 2:**
- Teaches: Distributed learning, privacy, advanced ML
- Requires: Understanding from V1
- Expected: 12-16 hours of active coding

**Total Investment:**
- 25-30 hours of focused coding
- Perfect for portfolio
- Professional quality results

---

## ✅ Final Checklist Before You Start

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Kaggle API configured
- [ ] Project structure created
- [ ] Claude.md files in correct locations
- [ ] VS Code with Claude Code extension installed
- [ ] Claude authenticated in VS Code
- [ ] Git initialized in project
- [ ] First commit made
- [ ] README created
- [ ] .gitignore in place
- [ ] Can open project in VS Code
- [ ] Claude Code reads claude.md (test with simple prompt)

---

## 🆘 If Something Goes Wrong

### Claude Code Not Reading claude.md
- Ensure file is named exactly `claude.md`
- Must be in root of opened folder
- Restart VS Code
- Try Ctrl+R to reload window

### Kaggle Dataset Download Fails
- Verify kaggle.json is in right location
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Try downloading manually from web

### Python Dependency Conflicts
- Create fresh virtual environment
- Install dependencies one by one
- Use specified versions in requirements.txt

### Claude Code Suggestions Seem Off
- Be more specific in your prompts
- Reference your claude.md file explicitly
- Provide more context
- Ask Claude to review against your guidelines

---

## 📞 Next Steps

1. **Complete Step-by-Step Setup** above (takes ~1-2 hours total)
2. **Open V1 folder** in VS Code
3. **Ask Claude first prompt** from CLAUDE_CODE_EXAMPLES.md
4. **Review and test** generated code
5. **Iterate and commit** as you progress
6. **Enjoy building** your portfolio project!

---

## 🎉 You're Ready!

You now have:
- ✅ Complete project specifications (claude.md files)
- ✅ Comprehensive documentation (PDF + guides)
- ✅ Code examples and prompts
- ✅ Setup instructions
- ✅ Development workflow
- ✅ GitHub-ready structure

Everything is in place to build a professional portfolio project with Claude Code assistance. The combination of clear specifications (claude.md) and AI assistance (Claude Code) will make your development smooth and your results professional.

**Happy coding! 🚀**

---

## 📚 Complete File Reference

| File | Purpose | When to Use |
|------|---------|------------|
| ML_Portfolio_Projects_Guide.pdf | Full project guide | Overview & detailed understanding |
| QUICK_REFERENCE.md | Quick lookup guide | During coding (commands, snippets) |
| V1_CLAUDE.md | V1 specification | Rename to `claude.md` in v1-basic-ml/ |
| V2_CLAUDE.md | V2 specification | Rename to `claude.md` in v2-federated-learning/ |
| CLAUDE_CODE_GUIDE.md | Using Claude Code | Learn about Claude Code extension |
| CLAUDE_CODE_EXAMPLES.md | Prompt examples | Generate V1 and V2 code |
| This file | Setup & workflow | Getting started |

Start with Step 1 in the setup section and work your way through. Good luck! 🚀
