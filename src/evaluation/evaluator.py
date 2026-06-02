import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model_performance(model, X_test, y_test, class_names, output_dir="outputs"):
    """
    Computes performance metrics and plots confusion matrix.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    t_X = torch.tensor(X_test, dtype=torch.float32)
    t_y = torch.tensor(y_test, dtype=torch.long)
    
    with torch.no_grad():
        logits = model(t_X)
        _, preds = torch.max(logits, 1)
        
    # Classification Report
    y_true_np = t_y.numpy()
    y_pred_np = preds.numpy()
    
    report = classification_report(y_true_np, y_pred_np, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true_np, y_pred_np, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    return report

def calculate_gradient_attribution(model, preprocessed_input, target_class_idx, feature_names):
    """
    Calculates feature attribution (saliency) for a single input using PyTorch gradients.
    Shows the relative sensitivity of the target class logit to each input feature.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        preprocessed_input (np.ndarray): 1D array of preprocessed features (shape: (input_dim,)).
        target_class_idx (int): The index of the predicted/target class.
        feature_names (list): List of feature names corresponding to the preprocessed columns.
    Returns:
        pd.DataFrame: Attribution values for each feature.
    """
    model.eval()
    
    # Convert input to tensor and enable gradient tracking
    input_tensor = torch.tensor(preprocessed_input, dtype=torch.float32).unsqueeze(0) # Shape: (1, input_dim)
    input_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(input_tensor)
    
    # Target logit
    target_logit = outputs[0, target_class_idx]
    
    # Backpropagate to compute gradients w.r.t input
    model.zero_grad()
    target_logit.backward()
    
    # Gradients represent attribution / sensitivity
    gradients = input_tensor.grad.detach().numpy()[0]
    
    # Simple input * gradient attribution (Saliency)
    # This measures the impact of the feature scale * its direction
    attributions = preprocessed_input * gradients
    
    attr_df = pd.DataFrame({
        "feature": feature_names,
        "attribution": attributions,
        "raw_gradient": gradients
    })
    
    # Sort by absolute attribution
    attr_df["abs_attr"] = attr_df["attribution"].abs()
    attr_df = attr_df.sort_values(by="abs_attr", ascending=False).reset_index(drop=True)
    
    return attr_df

def calculate_global_importance(model, preprocessed_dataset, feature_names):
    """
    Calculates global feature importance by averaging the absolute gradients
    of all samples in the dataset across all predicted classes.
    """
    model.eval()
    
    input_tensor = torch.tensor(preprocessed_dataset, dtype=torch.float32)
    input_tensor.requires_grad = True
    
    outputs = model(input_tensor)
    
    # Compute sum of max logits to do backprop once for efficiency
    max_logits, _ = torch.max(outputs, dim=1)
    sum_max_logits = max_logits.sum()
    
    model.zero_grad()
    sum_max_logits.backward()
    
    avg_abs_gradients = input_tensor.grad.abs().mean(dim=0).detach().numpy()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_abs_gradients
    })
    
    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return importance_df
