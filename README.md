# Multimodal AI Framework for Autism Spectrum Disorder Risk Screening

This repository contains a two-stage deep learning system for Autism Spectrum Disorder (ASD) classification using multimodal data. The system uses both text and audio data to classify children into different risk categories.

## Project Structure

```
├── model1/                 # Stage 1: TD vs (High-Risk + ASD) Classification
│   ├── model.py           # Main model architecture (722 lines)
│   ├── dataset.py         # Data loading and preprocessing (386 lines)
│   ├── train_deep_ensemble2.py  # Training script (173 lines)
│   └── notebooks/         # Jupyter notebooks for analysis
│       ├── model1_data preprocess_Final.ipynb  # Data preprocessing and analysis
│       └── test_model_text.ipynb              # Model testing and evaluation
│
├── model2/                # Stage 2: High-Risk vs ASD Classification
│   ├── model.py          # Stage 2 model architecture (78 lines)
│   ├── dataset.py        # Stage 2 data processing (57 lines)
│   ├── train.py          # Stage 2 training script (174 lines)
│   └── notebooks/        # Analysis and development notebooks
│       ├── 1_Data_Preprocessing/    # Data preparation and cleaning
│       │   └── model2_data preprocess_Final.ipynb  # Final data preprocessing pipeline
│       ├── 2_Model_Development/     # Model implementation and training
│       │   ├── model2_git2.finetune_roberta.ipynb           # Initial model fine-tuning
│       │   └── model2_git2.finetune_roberta_compared_seed42.ipynb  # Reproducibility study
│       └── 3_Model_Analysis/        # Model evaluation and analysis
│           └── model2_.Attribution Analysis_Final.ipynb  # Feature attribution analysis
│
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Stage 1: TD vs (High-Risk + ASD) Classification

This model distinguishes between typically developing (TD) children and those at high-risk for ASD or diagnosed with ASD using multimodal data.

### Features
- Multimodal input processing (text and audio)
- Text data from MCHAT and SCQ-L questionnaires
- Audio data from parent-child interaction tasks
- Multiple Instance Learning (MIL) framework
- RoBERTa-large for text processing
- Whisper model for audio feature extraction
- Auxiliary language delay prediction task
- Hard ensemble approach for final classification

### Dataset Classes
The `model1/dataset.py` implements several dataset classes:

1. `ASDDataset`: Basic audio dataset
2. `ASDWhisperDataset`: Audio with Whisper features
3. `ASDWhisperDatasetOverlap`: Overlapping segments
4. `ASDWhisperClinicalDataset`: Clinical data integration
5. `CustomRobertaDataset`: Multimodal processing
6. `CustomRobertaNoisyAudioDataset`: Robustness testing
7. `MultimodalASDDataset`: Main dataset class

### Dataset Statistics
- Total participants: 818 children
  - 273 TD children
  - 175 High-Risk children
  - 370 ASD children

## Stage 2: High-Risk vs ASD Classification

This model further classifies children into High-Risk or ASD categories. See [model2/README.md](model2/README.md) for detailed documentation.

### Features
- Integration of parent-child interaction task outcomes
- SRS-2 questionnaire responses
- RoBERTa Large model fine-tuning
- Binary classification for task success/failure
- AUC performance: 0.93

### Workflow
1. Data Preprocessing
   - Initial data cleaning and validation
   - Advanced feature engineering
   - Cross-validation split generation

2. Model Development
   - RoBERTa fine-tuning implementation
   - Hyperparameter optimization
   - Reproducibility analysis

3. Model Analysis
   - Feature attribution analysis
   - Uncertainty quantification
   - Error analysis

4. Results Analysis
   - Performance metrics visualization
   - Statistical significance testing
   - Cross-validation results


```

## Usage

### Stage 1 Model
1. Prepare your data:
   - MCHAT and SCQ-L questionnaire responses
   - Audio recordings from parent-child interactions
   - Language delay assessment data

2. Run training:
```bash
python model1/train_deep_ensemble2.py
```

3. Use the model for prediction:
```python
from model1.model import MultimodalASDClassifier
from model1.dataset import MultimodalASDDataset

model = MultimodalASDClassifier.from_pretrained('roberta-large', num_labels=2)
predictions = model.predict(text_data, audio_data)
```

### Stage 2 Model
1. Follow the notebooks in sequence:
   - Data preprocessing notebooks
   - Model development notebooks
   - Analysis notebooks

2. Run training:
```bash
python model2/train.py
```

3. Use the model for prediction:
```python
from model2.model import Stage2ASDClassifier
from model2.dataset import Stage2ASDDataset

model = Stage2ASDClassifier.from_pretrained('roberta-large', num_labels=2)
predictions = model.predict(text_data)
```

