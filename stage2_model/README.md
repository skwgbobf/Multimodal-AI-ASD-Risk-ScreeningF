# Stage 2: High-Risk vs ASD Classification

This directory contains the implementation of the second stage classifier that distinguishes between High-Risk and ASD cases.

## Model Architecture

- Text processing using RoBERTa-large
- Custom classification head with dropout and tanh activation
- Binary classification (High-Risk vs ASD)

## Project Structure

```
stage2_model/
├── model.py              # Stage 2 model architecture (78 lines)
├── dataset.py            # Stage 2 data processing (57 lines)
├── train.py              # Stage 2 training script (174 lines)
├── README.md             # This file
└── notebooks/            # Analysis notebooks
    ├── 1_Data_Preprocessing/    # Data preparation and cleaning
    │   └── model2_data preprocess_Final.ipynb  # Final data preprocessing pipeline
    ├── 2_Model_Development/     # Model implementation and training
    │   └── model2_git2.finetune_roberta.ipynb  # RoBERTa fine-tuning implementation
    └── 3_Model_Analysis/        # Model evaluation and analysis
        └── model2_.Attribution Analysis_Final.ipynb  # Feature attribution analysis
```

## Notebook Workflow

### 1. Data Preprocessing (`1_Data_Preprocessing/`)
The `model2_data preprocess_Final.ipynb` notebook handles:
- Loading and merging SRS data and interaction data
- Processing task results (name response, mimicked actions, played catch, etc.)
- Creating success/failure indicators for each task
- Combining mimicked actions data
- Merging interaction data with SRS scores
- Saving processed data in JSON format

### 2. Model Development (`2_Model_Development/`)
The `model2_git2.finetune_roberta.ipynb` notebook implements:
- Data preprocessing for RoBERTa fine-tuning
- Label mapping and class balancing
- Text formatting for model input
- Model training setup and execution
- Performance evaluation
- Model saving and checkpointing

### 3. Model Analysis (`3_Model_Analysis/`)
The `model2_.Attribution Analysis_Final.ipynb` notebook performs:
- Model evaluation and performance metrics calculation
- Expected Calibration Error (ECE) analysis
- Feature attribution analysis
- Model uncertainty quantification
- Performance visualization
- Cross-validation results analysis

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

## Usage

1. Prepare your data:
   - Parent-child interaction task outcomes
   - SRS-2 questionnaire responses
   - Clinical assessment data

2. Run training:
```bash
python train.py
```

3. Use the model for prediction:
```python
from model import Stage2ASDClassifier
from dataset import Stage2ASDDataset

model = Stage2ASDClassifier.from_pretrained('roberta-large', num_labels=2)
predictions = model.predict(text_data)
```

## Data Requirements

The model requires the following data structure:
```
data/
├── questionnaires/    # Questionnaire responses
│   └── srs2/         # SRS-2 responses
└── clinical/         # Clinical assessment data
    └── diagnosis/    # Clinical diagnoses
```

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.15.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- Other dependencies (see main requirements.txt) 