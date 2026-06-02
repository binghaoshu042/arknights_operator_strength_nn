import torch
import torch.nn as nn

class OperatorStrengthNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=4, dropout_rate=0.2):
        """
        Multi-Layer Perceptron (MLP) for Operator Strength Classification.
        
        Args:
            input_dim (int): Number of input features after encoding/scaling.
            hidden_dims (list): List of sizes for hidden layers.
            output_dim (int): Number of target classes (4: 小杯, 中杯, 大杯, 超大杯).
            dropout_rate (float): Dropout probability for regularization.
        """
        super(OperatorStrengthNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers dynamically
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
            
        # Output layer (probabilities are derived via CrossEntropyLoss, so we output raw logits)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Feature tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Logit tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
