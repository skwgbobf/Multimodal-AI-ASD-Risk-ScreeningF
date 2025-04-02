# Stage 1: TD vs (High-Risk + ASD) Classification

This directory contains the implementation of the first stage classifier that distinguishes between typically developing (TD) children and those at high-risk for ASD or diagnosed with ASD using multimodal data.

## Model Architecture

The model consists of two main pathways:

### Text Pathway
- Uses RoBERTa-large for processing questionnaire data
- Processes MCHAT and SCQ-L responses
- Implements a mapping strategy with 1,943 medical concepts
- Maps to 3,336 ASD-related terms
- Uses Sentence Transformer for cosine similarity scoring
- Selects top 5 matching terms with clinical relevance

### Audio Pathway
- Processes 3-minute parent-child interaction recordings
- Segments audio into 30-second intervals
- Uses Whisper model as feature extraction backbone
- Implements Multiple Instance Learning (MIL) framework
- Includes language delay prediction as auxiliary task

### Fusion and Classification
- Concatenates text and audio features
- Implements hard ensemble approach
- Combines predictions from:
  - MCHAT/SCQ-L based classification
  - Language delay prediction model

## Project Structure

```
Stage1_model/
├── model.py              # Main model architecture (722 lines)
├── dataset.py            # Data loading and preprocessing (386 lines)
├── train_deep_ensemble2.py  # Training script (173 lines)
├── README.md             # This file
└── notebooks/            # Analysis notebooks
    └── Stage1_model_text.ipynb  # Text model analysis
```

## Dataset Classes

The `dataset.py` implements several dataset classes:

1. `ASDDataset`: Basic audio dataset
2. `ASDWhisperDataset`: Audio with Whisper features
3. `ASDWhisperDatasetOverlap`: Overlapping segments
4. `ASDWhisperClinicalDataset`: Clinical data integration
5. `CustomRobertaDataset`: Multimodal processing
6. `CustomRobertaNoisyAudioDataset`: Robustness testing
7. `MultimodalASDDataset`: Main dataset class

## Dataset Statistics

- Total participants: 818 children
  - 273 TD children
  - 175 High-Risk children
  - 370 ASD children

## Training Process

The model uses a composite loss function:
- Binary cross-entropy for main classification
- MSE loss for language delay prediction
- Weighted combination of both losses

## Usage

1. Prepare your data:
   - MCHAT and SCQ-L questionnaire responses
   - Audio recordings from parent-child interactions
   - Language delay assessment data

2. Run training:
```bash
python train_deep_ensemble2.py
```

3. Use the model for prediction:
```python
from model import MultimodalASDClassifier
from dataset import MultimodalASDDataset

model = MultimodalASDClassifier.from_pretrained('roberta-large', num_labels=2)
predictions = model.predict(text_data, audio_data)
```

## Data Requirements

The model expects the following data structure:
```
data/
├── questionnaires/    # Questionnaire responses
│   ├── mchat/        # MCHAT responses
│   └── scql/         # SCQ-L responses
├── audio/            # Audio recordings
│   └── interactions/ # Parent-child interaction recordings
└── clinical/         # Clinical assessment data
    └── language/     # Language delay assessments
```

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.15.0
- Torchaudio >= 0.9.0
- OpenAI Whisper >= 20231117
- Sentence Transformers >= 2.2.0
- Other dependencies (see main requirements.txt) 