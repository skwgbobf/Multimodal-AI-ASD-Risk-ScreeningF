# Stage 1: TD vs (High-Risk + ASD) Classification Model

This model implements a multimodal classification system to distinguish between typically developing (TD) children and those at high-risk for ASD or diagnosed with ASD.

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

## Files

- `model.py`: Main model architecture implementation
- `dataset.py`: Data loading and preprocessing utilities
- `train_deep_ensemble2.py`: Training script with ensemble approach
- `notebooks/`: Analysis and visualization notebooks

## Training

The model uses a composite loss function:
- Classification loss (Cross-Entropy)
- Language delay prediction loss (Cross-Entropy)
- Total loss = average of component losses

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
   from model import MultimodalClassifier
   
   model = MultimodalClassifier()
   predictions = model.predict(text_data, audio_data)
   ```

## Requirements
- PyTorch
- Transformers
- Whisper
- Other dependencies (see requirements.txt) 