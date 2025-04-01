"""
Dataset classes for Stage 2 ASD Classification
This module implements custom dataset classes for processing text data for Stage 2 ASD classification.
"""

from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizerFast

class Stage2ASDDataset(Dataset):
    """Dataset class for Stage 2 ASD classification (High-Risk vs ASD)."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing the dataset with columns:
                - text: Text data from questionnaires
                - Class_x: Class labels ('High-Risk' or 'ASD')
        """
        self.df = df
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized input IDs
                - attention_mask: Attention mask
                - labels: Binary labels (0 for High-Risk, 1 for ASD)
        """
        row = self.df.iloc[idx]
        
        # Process text
        inputs = self.tokenizer(
            row.text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=514
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Prepare label (1 for ASD, 0 for High-Risk)
        label = 1 if row.Class_x == 'ASD' else 0
        inputs['labels'] = label
        
        return inputs 