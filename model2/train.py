"""
Training script for Stage 2 ASD Classification Model
This script implements the training loop for the Stage 2 classifier.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn

from dataset import Stage2ASDDataset
from model import Stage2ASDClassifier

def set_seed(seed=41):
    """Set random seeds for reproducibility."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def train(fold, data_dir='./data2', output_dir='./results'):
    """
    Train the Stage 2 ASD classifier.
    
    Args:
        fold (int): Current fold number for cross-validation
        data_dir (str): Directory containing the dataset
        output_dir (str): Directory to save model checkpoints and logs
    """
    # Set up output directory
    os.makedirs(f'{output_dir}/fold{fold}', exist_ok=True)
    sys.stdout = open(f'{output_dir}/fold{fold}/log.txt', 'w')

    # Load data
    test = pd.read_csv(f'{data_dir}/test.csv')
    val = pd.read_csv(f'{data_dir}/fold_{fold}/val.csv')
    train = pd.read_csv(f'{data_dir}/fold_{fold}/train.csv')

    # Filter for High-Risk and ASD cases only
    train = train[train['Class_x'].isin(['High-Risk', 'ASD'])]
    val = val[val['Class_x'].isin(['High-Risk', 'ASD'])]
    test = test[test['Class_x'].isin(['High-Risk', 'ASD'])]

    # Create datasets
    train_dataset = Stage2ASDDataset(train)
    val_dataset = Stage2ASDDataset(val)
    test_dataset = Stage2ASDDataset(test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage2ASDClassifier.from_pretrained('roberta-large', num_labels=2)
    model = model.to(device)
    softmax = nn.Softmax(dim=1)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 10
    best_valid_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for input in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            input = {key: val.to(device) for key, val in input.items()}
            optimizer.zero_grad()
            
            output = model(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                labels=input['labels']
            )
            
            loss = output.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        pred_list = []
        true_list = []
        valid_loss = 0.0
        
        with torch.no_grad():
            for input in tqdm(val_dataloader, desc='Validation'):
                input = {key: val.to(device) for key, val in input.items()}
                output = model(
                    input_ids=input['input_ids'],
                    attention_mask=input['attention_mask'],
                    labels=input['labels']
                )
                
                valid_loss += output.loss.item()
                pred_list.append(softmax(output.logits).cpu().numpy()[:, 1])
                true_list.append(input['labels'].cpu().numpy())
        
        valid_loss /= len(val_dataloader)
        valid_auc = roc_auc_score(np.concatenate(true_list), np.concatenate(pred_list))
        valid_acc = accuracy_score(np.concatenate(true_list), np.concatenate(pred_list) > 0.5)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'{output_dir}/fold{fold}/best_model.pt')

        # Log metrics
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'\tTrain Loss: {train_loss:.5f}')
        print(f'\tValid Loss: {valid_loss:.5f}')
        print(f'\tValid AUC: {valid_auc:.5f}')
        print(f'\tValid ACC: {valid_acc:.5f}')

    # Test phase
    model.load_state_dict(torch.load(f'{output_dir}/fold{fold}/best_model.pt'))
    model.eval()
    
    pred_list = []
    true_list = []
    test_loss = 0.0
    
    with torch.no_grad():
        for input in tqdm(test_dataloader, desc='Testing'):
            input = {key: val.to(device) for key, val in input.items()}
            output = model(
                input_ids=input['input_ids'],
                attention_mask=input['attention_mask'],
                labels=input['labels']
            )
            
            test_loss += output.loss.item()
            pred_list.append(softmax(output.logits).cpu().numpy()[:, 1])
            true_list.append(input['labels'].cpu().numpy())
    
    test_loss /= len(test_dataloader)
    test_auc = roc_auc_score(np.concatenate(true_list), np.concatenate(pred_list))
    test_acc = accuracy_score(np.concatenate(true_list), np.concatenate(pred_list) > 0.5)

    # Log final results
    print(f'\nFinal Results:')
    print(f'\tTest Loss: {test_loss:.5f}')
    print(f'\tTest AUC: {test_auc:.5f}')
    print(f'\tTest ACC: {test_acc:.5f}')

    sys.stdout.close()

if __name__ == '__main__':
    set_seed()
    train(fold=4)  # Example: train on fold 4 