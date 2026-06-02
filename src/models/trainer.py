import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from src.data.data_loader import load_profession_data
from src.features.feature_extractor import OperatorFeatureExtractor
from src.models.nn_model import OperatorStrengthNN

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, epochs=150, lr=0.002, weight_decay=1e-4, patience=15, device="cpu"):
    """
    Core PyTorch training loop with Early Stopping.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    best_loss = float("inf")
    best_state = None
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # Save best state dict
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_state = model.state_dict().copy()
            
        # Check early stopping
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            # print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(best_state)
    return model, history

def run_profession_pipeline(profession, data_path="data/raw/operators.csv", epochs=150, batch_size=32, device="cpu"):
    """
    Runs the full workflow for a specific operator profession:
    - Load specific data
    - Feature extract
    - Train neural network
    - Evaluate on test set
    - Save model & preprocessor
    """
    print(f"\n==================================================")
    print(f"Starting Training Pipeline for Profession: {profession.upper()}")
    print(f"==================================================")
    
    # 1. Load Data
    X_train_df, X_test_df, y_train_series, y_test_series = load_profession_data(
        file_path=data_path, profession=profession
    )
    print(f"Data Loaded: {len(X_train_df)} training samples, {len(X_test_df)} test samples.")
    
    # 2. Extract Features
    extractor = OperatorFeatureExtractor()
    X_train_arr = extractor.fit_transform(X_train_df)
    X_test_arr = extractor.transform(X_test_df)
    
    y_train_arr = extractor.encode_labels(y_train_series)
    y_test_arr = extractor.encode_labels(y_test_series)
    
    print(f"Feature Dimension: {extractor.get_input_dim()} inputs")
    
    # 3. Create DataLoaders
    # We use 10% of training data as validation set
    val_split = int(0.1 * len(X_train_arr))
    if val_split == 0:
        val_split = 1
        
    X_val_arr = X_train_arr[:val_split]
    y_val_arr = y_train_arr[:val_split]
    
    X_train_real_arr = X_train_arr[val_split:]
    y_train_real_arr = y_train_arr[val_split:]
    
    # Convert to Tensors
    t_X_train = torch.tensor(X_train_real_arr, dtype=torch.float32)
    t_y_train = torch.tensor(y_train_real_arr, dtype=torch.long)
    t_X_val = torch.tensor(X_val_arr, dtype=torch.float32)
    t_y_val = torch.tensor(y_val_arr, dtype=torch.long)
    t_X_test = torch.tensor(X_test_arr, dtype=torch.float32)
    t_y_test = torch.tensor(y_test_arr, dtype=torch.long)
    
    train_dataset = TensorDataset(t_X_train, t_y_train)
    val_dataset = TensorDataset(t_X_val, t_y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Neural Network
    model = OperatorStrengthNN(input_dim=extractor.get_input_dim())
    
    # 5. Train Model
    trained_model, history = train_model(
        model, train_loader, val_loader, epochs=epochs, device=device
    )
    
    # 6. Evaluate Model on Test Set
    trained_model.eval()
    with torch.no_grad():
        test_inputs = t_X_test.to(device)
        test_outputs = trained_model(test_inputs)
        _, preds = torch.max(test_outputs, 1)
        corrects = (preds == t_y_test.to(device)).sum().item()
        test_acc = corrects / len(t_y_test)
        
    print(f"Training Complete! Test Accuracy: {test_acc:.4%}")
    
    # 7. Save Model & Preprocessor
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_dir, f"{profession.lower()}_model.pth")
    preprocessor_save_path = os.path.join(model_dir, f"{profession.lower()}_preprocessor.pkl")
    
    torch.save(trained_model.state_dict(), model_save_path)
    extractor.save(preprocessor_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print(f"Preprocessor saved to: {preprocessor_save_path}")
    
    return {
        "profession": profession,
        "test_acc": test_acc,
        "history": history
    }

def run_unified_pipeline(data_path="data/raw/operators.csv", epochs=150, batch_size=32, device="cpu"):
    """
    Runs the full training workflow for the UNIFIED operator strength model:
    - Load all operator data
    - Feature extract (with profession, type, branch encoded)
    - Train a single unified neural network
    - Evaluate on test set
    - Save unified model & preprocessor
    """
    print(f"\n==================================================")
    print(f"Starting Training Pipeline for UNIFIED Model")
    print(f"==================================================")
    
    # 1. Load Data (profession=None gets all data)
    X_train_df, X_test_df, y_train_series, y_test_series = load_profession_data(
        file_path=data_path, profession=None
    )
    print(f"Data Loaded: {len(X_train_df)} training samples, {len(X_test_df)} test samples.")
    
    # 2. Extract Features
    extractor = OperatorFeatureExtractor()
    X_train_arr = extractor.fit_transform(X_train_df)
    X_test_arr = extractor.transform(X_test_df)
    
    y_train_arr = extractor.encode_labels(y_train_series)
    y_test_arr = extractor.encode_labels(y_test_series)
    
    print(f"Unified Feature Dimension: {extractor.get_input_dim()} inputs")
    
    # 3. Create DataLoaders
    val_split = int(0.1 * len(X_train_arr))
    if val_split == 0:
        val_split = 1
        
    X_val_arr = X_train_arr[:val_split]
    y_val_arr = y_train_arr[:val_split]
    
    X_train_real_arr = X_train_arr[val_split:]
    y_train_real_arr = y_train_arr[val_split:]
    
    # Convert to Tensors
    t_X_train = torch.tensor(X_train_real_arr, dtype=torch.float32)
    t_y_train = torch.tensor(y_train_real_arr, dtype=torch.long)
    t_X_val = torch.tensor(X_val_arr, dtype=torch.float32)
    t_y_val = torch.tensor(y_val_arr, dtype=torch.long)
    t_X_test = torch.tensor(X_test_arr, dtype=torch.float32)
    t_y_test = torch.tensor(y_test_arr, dtype=torch.long)
    
    train_dataset = TensorDataset(t_X_train, t_y_train)
    val_dataset = TensorDataset(t_X_val, t_y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Neural Network (wider hidden layer for unified model)
    model = OperatorStrengthNN(input_dim=extractor.get_input_dim(), hidden_dims=[256, 128, 64])
    
    # 5. Train Model
    trained_model, history = train_model(
        model, train_loader, val_loader, epochs=epochs, device=device
    )
    
    # 6. Evaluate Model on Test Set
    trained_model.eval()
    with torch.no_grad():
        test_inputs = t_X_test.to(device)
        test_outputs = trained_model(test_inputs)
        _, preds = torch.max(test_outputs, 1)
        corrects = (preds == t_y_test.to(device)).sum().item()
        test_acc = corrects / len(t_y_test)
        
    print(f"Training Complete! Unified Test Accuracy: {test_acc:.4%}")
    
    # 7. Save Model & Preprocessor
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_dir, "unified_model.pth")
    preprocessor_save_path = os.path.join(model_dir, "unified_preprocessor.pkl")
    
    torch.save(trained_model.state_dict(), model_save_path)
    extractor.save(preprocessor_save_path)
    
    print(f"Unified Model saved to: {model_save_path}")
    print(f"Unified Preprocessor saved to: {preprocessor_save_path}")
    
    return {
        "profession": "Unified",
        "test_acc": test_acc,
        "history": history
    }
