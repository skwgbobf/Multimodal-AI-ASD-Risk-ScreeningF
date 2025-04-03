# Multimodal AI Framework for Autism Spectrum Disorder Risk Screening

This repository contains a two-stage deep learning system for Autism Spectrum Disorder (ASD) classification using multimodal data. The system uses both text and audio data to classify children into different risk categories.

## Project Structure

```
.
├── Stage1_model/              # Stage 1: TD vs (High-Risk + ASD) Classification
│   ├── model.py              # Main model architecture 
│   ├── dataset.py            # Data loading and preprocessing 
│   ├── train_deep_ensemble2.py  # Training script 
│   └── notebooks/            # Analysis notebooks
│       └── Stage1_model_text.ipynb  # Text model analysis
│
├── stage2_model/             # Stage 2: High-Risk vs ASD Classification
│   ├── model.py              # Stage 2 model architecture 
│   ├── dataset.py            # Stage 2 data processing 
│   ├── train.py              # Stage 2 training script 
    └── notebooks/            # Analysis notebooks
│       ├── 1_Data_Preprocessing/    # Data preparation and cleaning
│       │   └── model2_data preprocess_Final.ipynb  # Final data preprocessing pipeline
│       ├── 2_Model_Development/     # Model implementation and training
│       │   └── Stage2_model_finetune_roberta.ipynb  # RoBERTa fine-tuning implementation
│       └── 3_Model_Analysis/        # Model evaluation and analysis
│           └── Stage2model_Analysis_Final.ipynb  # Model application analysis
│
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Overview

This project implements a two-stage classification system for ASD risk screening:

### Stage 1: TD vs (High-Risk + ASD) Classification
- Distinguishes between typically developing (TD) children and those at high-risk for ASD or diagnosed with ASD
- Uses multimodal data (text and audio)
- Implements Multiple Instance Learning (MIL) framework
- Combines RoBERTa-large for text processing and Whisper for audio feature extraction

### Stage 2: High-Risk vs ASD Classification
- Further classifies children into High-Risk or ASD categories
- Uses RoBERTa-large for text processing
- Focuses on parent-child interaction task outcomes and SRS-2 questionnaire responses


## Data Preprocessing

The project includes two main preprocessing pipelines:

1. Main Preprocessing (`1.Preprocess_Final.ipynb`):
   - Comprehensive data cleaning and validation
   - Feature extraction from questionnaires
   - Advanced feature engineering
   - Cross-validation split generation
   - Feature importance analysis
   - Data distribution analysis
   - Missing value handling
   - Outlier detection

2. Stage 2 Specific Preprocessing (`stage2_model/notebooks/1_Data_Preprocessing/`):
   - SRS data and interaction data merging
   - Task result processing
   - Success/failure indicators
   - Data formatting for model input

## Model Development

### Stage 1 Model
- Multimodal architecture combining text and audio processing
- Multiple dataset classes for different data types
- Ensemble approach for final classification


### Stage 2 Model
- RoBERTa-large based text classification
- Binary classification (High-Risk vs ASD)
- Training parameters:
  - Epochs: 10
  - Batch size: 8
  - Learning rate: 2e-5
  - Optimizer: AdamW with weight decay 0.01

## Model Analysis

The project includes comprehensive analysis tools:

1. Performance Metrics:
   - AUC, Accuracy, Precision, Recall
   - Expected Calibration Error (ECE)
   - Cross-validation results

2.  Analysis:
   - Model uncertainty quantification
   - Performance visualization
   - ADOS relatioship and severity stratificaitn 


## Usage

1. Data Preprocessing:
   - Run `1.Preprocess_Final.ipynb` for main preprocessing
   - Follow stage-specific preprocessing notebooks

2. Stage 1 Model:
   ```python
   from Stage1_model.model import MultimodalASDClassifier
   from Stage1_model.dataset import MultimodalASDDataset
   
   model = MultimodalASDClassifier.from_pretrained('roberta-large', num_labels=2)
   predictions = model.predict(text_data, audio_data)
   ```

3. Stage 2 Model:
   ```python
   from stage2_model.model import Stage2ASDClassifier
   from stage2_model.dataset import Stage2ASDDataset
   
   model = Stage2ASDClassifier.from_pretrained('roberta-large', num_labels=2)
   predictions = model.predict(text_data)
   ```

## Data Requirements

The project requires the following data structure:

```
data/
├── questionnaires/              # Questionnaire responses
│   ├── mchat/                  # MCHAT responses (Stage 1)
│   ├── scq_l/                  # SCQ-L responses (Stage 1)
│   └── srs2/                   # SRS-2 responses (Stage 2)
│
├── audio/                      # Audio recordings from parent-child interactions
│   ├── free_play/             # Free Play task recordings (Stage 1)
│   │   └── 3min_segments/     # 30-second segments
│   └── interaction_tasks/     # 5 standardized tasks (Stage 2)
│       ├── task1/            # Task recordings
│       ├── task2/
│       ├── task3/
│       ├── task4/
│       └── task5/            # (6 tasks for 36-48 months)
│
├── clinical/                   # Clinical assessment data
│   ├── ados/                  # ADOS-2 assessment data
│   │   ├── mod1/             # Module 1
│   │   ├── mod2/             # Module 2
│   │   └── modt/             # Toddler Module
│   │       ├── ados_sa/      # Social Affect scores
│   │       └── ados_rrb/     # Restricted/Repetitive Behavior scores
│   └── language/             # Language assessment data
│       ├── pres/             # Preschool Receptive-Expressive Language Scale
│       └── selsi/            # Sequenced Language Scale for Infants
│
└── metadata/                  # Additional participant information
    ├── demographics/         # Age, gender, etc.
    ├── exclusion_criteria/   # Family history, premature birth info
    └── language_delay/       # Language delay classification
```

### Dataset Statistics

#### Stage 1 Dataset (TD vs High-Risk + ASD)
- Initial Sample: 1,242 participants
  - 434 TD children
  - 331 High-Risk children
  - 477 ASD children
- Final Dataset: 818 children
  - 273 TD children
  - 175 High-Risk children
  - 370 ASD children
- Audio Data: 897 children
  - 294 TD children
  - 214 High-Risk children
  - 389 ASD children

#### Stage 2 Dataset (High-Risk vs ASD)
- Total Sample: 515 children
  - 162 High-Risk children
  - 353 ASD children
- Audio Data: 547 children

### Data Collection Details

1. **Survey Data**
   - MCHAT and SCQ-L: Used for Stage 1 classification
   - SRS-2: Used for Stage 2 classification
   - Text-based data processed using RoBERTa-large

2. **Digital Phenotyping Data**
   - Parent-child interaction tasks (7 variations)
   - Task count varies by age:
     - 18-23 months: 4 tasks
     - 24-35 months: 5 tasks
     - 36-48 months: 6 tasks
   - Audio recordings processed using Whisper model

3. **Clinical Data**
   - ADOS-2 assessments (Modules 1, 2, and Toddler)
   - Composite ADOS-2 total(T) score calculation:
     - ADOS-2 TOTAL(T) = ADOS_SA + ADOS_RRB
   - Language delay assessment:
     - Delay threshold: 7 months
     - Assessments: PRES or SELSI
     - Classification based on receptive/expressive language age

### Data Preprocessing Requirements

1. **Text Data Processing**
   - Mapping of 1,943 medical concepts to 3,336 ASD-related terms
   - Sentence Transformer for cosine similarity calculation
   - Keyword selection based on clinical judgment

2. **Audio Data Processing**
   - 3-minute recordings segmented into 30-second intervals
   - Multiple Instance Learning (MIL) framework implementation
   - Whisper model for feature extraction

3. **Clinical Data Processing**
   - ADOS-2 score normalization across modules
   - Language delay classification
   - Exclusion criteria application

## Dependencies

- Python 3.x
- PyTorch >= 1.9.0
- Transformers >= 4.15.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- Torchaudio >= 0.9.0
- OpenAI Whisper >= 20231117
- Sentence Transformers >= 2.2.0
- Other dependencies (see requirements.txt)

