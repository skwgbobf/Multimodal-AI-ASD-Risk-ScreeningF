# Stage 2: High-Risk vs ASD Classification

This directory contains the implementation of the second stage classifier that distinguishes between High-Risk and ASD cases.

## Model Architecture

The Stage 2 classifier is based on RoBERTa-large and includes:
- Text processing using RoBERTa-large
- Custom classification head with dropout and tanh activation
- Binary classification (High-Risk vs ASD)

## Project Structure

```
model2/
├── model.py              # Main model implementation
├── dataset.py           # Dataset processing classes
├── train.py            # Training script
└── notebooks/          # Analysis and development notebooks
    ├── 1_Data_Preprocessing/
    │   ├── model2_git1.data preprocess.ipynb        # Initial data preprocessing
    │   └── 1.Preprocess_20250103_F (2).ipynb       # Comprehensive preprocessing pipeline
    │
    ├── 2_Model_Development/
    │   ├── model2_git2.finetune_roberta.ipynb      # RoBERTa fine-tuning implementation
    │   └── model2_git2.finetune_roberta_compared_seed42.ipynb  # Seed comparison study
    │
    ├── 3_Model_Analysis/
    │   ├── model2_git3.Attribution AnalysisFF.ipynb  # Initial attribution analysis
    │   ├── model2_git3.Attribution AnalysisFF_new_data_Jan15F.ipynb  # Updated attribution analysis
    │   └── model2_git3.model_uncertainty_1004_result_nov26F_Jan14F.ipynb  # Uncertainty analysis
    │
    └── 4_Results_Analysis/
        └── model2_git3_data flow_model_uncertainty_result_comparisonF.ipynb  # Results comparison
```

## Notebook Workflow

### 1. Data Preprocessing (`1_Data_Preprocessing/`)
1. Initial Preprocessing (`model2_git1.data preprocess.ipynb`)
   - Data cleaning and validation
   - Feature extraction from questionnaires
   - Basic data quality checks
   - Initial data visualization

2. Comprehensive Preprocessing (`1.Preprocess_20250103_F (2).ipynb`)
   - Advanced feature engineering
   - Data augmentation techniques
   - Cross-validation split generation
   - Feature importance analysis
   - Data distribution analysis
   - Missing value handling
   - Outlier detection

### 2. Model Development (`2_Model_Development/`)
1. RoBERTa Fine-tuning (`model2_git2.finetune_roberta.ipynb`)
   - Model architecture implementation
   - Hyperparameter optimization
   - Training pipeline setup
   - Model checkpointing
   - Learning rate scheduling
   - Early stopping implementation
   - Model validation

2. Seed Comparison Study (`model2_git2.finetune_roberta_compared_seed42.ipynb`)
   - Reproducibility analysis
   - Model stability assessment
   - Performance comparison across seeds
   - Statistical significance testing
   - Error analysis

### 3. Model Analysis (`3_Model_Analysis/`)
1. Feature Attribution Analysis
   - Initial Analysis (`model2_git3.Attribution AnalysisFF.ipynb`)
     - Feature importance calculation
     - Attention visualization
     - Input contribution analysis
     - Token-level analysis
   
   - Updated Analysis (`model2_git3.Attribution AnalysisFF_new_data_Jan15F.ipynb`)
     - Enhanced visualization techniques
     - Comparative analysis with previous results
     - New feature importance metrics
     - Updated token analysis

2. Uncertainty Analysis (`model2_git3.model_uncertainty_1004_result_nov26F_Jan14F.ipynb`)
   - Model uncertainty quantification
   - Confidence score analysis
   - Error analysis and failure cases
   - Calibration analysis
   - Reliability diagrams
   - Uncertainty visualization

### 4. Results Analysis (`4_Results_Analysis/`)
1. Comprehensive Results (`model2_git3_data flow_model_uncertainty_result_comparisonF.ipynb`)
   - Performance metrics visualization
   - Model behavior analysis
   - Cross-validation results
   - Statistical significance testing
   - Error analysis
   - Results comparison across different approaches

## Usage

1. Data Preparation:
   ```python
   # Follow the preprocessing notebooks in order:
   # 1. model2_git1.data preprocess.ipynb
   # 2. 1.Preprocess_20250103_F (2).ipynb
   ```

2. Model Training:
   ```python
   # Use the training script
   python train.py
   ```

3. Model Analysis:
   ```python
   # Run the analysis notebooks in sequence:
   # 1. Feature attribution analysis
   # 2. Uncertainty analysis
   # 3. Results comparison
   ```

## Training Parameters

- Epochs: 10
- Batch size: 8
- Learning rate: 2e-5
- Optimizer: AdamW with weight decay 0.01
- Loss function: Cross-entropy
- Early stopping based on validation loss

## Performance

The model achieves:
- AUC: 0.93
- Accuracy: [To be added]
- Precision: [To be added]
- Recall: [To be added]

## Dependencies

- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Other dependencies (see main requirements.txt) 