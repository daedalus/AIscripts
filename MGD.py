#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metagradient Descent (MGD) Implementation in PyTorch

Implementation of the Metagradient Descent algorithm from the paper:
"Optimizing ML Training with Metagradient Descent" by Engstrom et al.
https://arxiv.org/pdf/2503.13751

Key Features:
- Computes gradients through the entire training process (metagradients)
- Supports optimization of metaparameters (data weights, hyperparameters)
- Efficient checkpointing system for memory management
- Modular design for easy extension

Author: Dario Clavijo
Date: Mar 26 2025 
License: MIT License

Usage Example:
    # Initialize model and data loaders
    model = SimpleModel()
    train_loader, val_loader = get_data_loaders()
    
    # Initialize MGD optimizer (z could be data weights, hyperparameters, etc.)
    mgd = MGDOptimizer(model, train_loader, val_loader, initial_z)
    
    # Run MGD optimization
    for epoch in range(num_epochs):
        z = mgd.step(n_steps=10)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

class SimpleModel(nn.Module):
    """A simple neural network for demonstration purposes
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Size of hidden layer
        output_dim: Number of output classes
    """
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class MGDOptimizer:
    """Main MGD optimizer class that implements the metagradient descent algorithm
    
    Args:
        model: PyTorch model to optimize
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        initial_z: Initial metaparameters (e.g., data weights)
        lr: Learning rate for metaparameter updates
    """
    def __init__(self, model, train_loader, val_loader, initial_z, lr=0.1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.z = initial_z
        self.lr = lr
        self.checkpoint_interval = 5  # Store state every N steps
    
    def weighted_loss(self, y_pred, y, z):
        """Compute loss weighted by metaparameters z
        
        Args:
            y_pred: Model predictions
            y: Ground truth labels
            z: Metaparameters (weights)
            
        Returns:
            Weighted loss value
        """
        loss = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
        return torch.mean(loss * z)
    
    def train_with_checkpoints(self, n_steps):
        """Train model while storing checkpoints of intermediate states
        
        Args:
            n_steps: Total number of training steps
            
        Returns:
            List of saved states (checkpoints)
        """
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        states = []
        
        for step, (x, y) in enumerate(self.train_loader):
            if step >= n_steps:
                break
                
            # Forward pass with metaparameter-dependent loss
            y_pred = self.model(x)
            loss = self.weighted_loss(y_pred, y, self.z)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save checkpoint periodically
            if step % self.checkpoint_interval == 0:
                states.append({
                    'step': step,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                })
        
        return states
    
    def gradient_of_phi_wrt_final_state(self):
        """Compute gradient of validation loss w.r.t. final model parameters
        
        Returns:
            Gradient tensor (same shape as model parameters)
        """
        x_val, y_val = next(iter(self.val_loader))
        y_pred = self.model(x_val)
        phi = nn.CrossEntropyLoss()(y_pred, y_val)
        
        # Compute ∇ϕ(θ_T)
        grad_phi = torch.autograd.grad(phi, self.model.parameters())
        return grad_phi
    
    def compute_gradient_through_step(self, A_t, state_t, batch):
        """Compute gradient contribution of step t to metagradient
        
        Args:
            A_t: Gradient from step t+1 (∂ϕ/∂s_{t+1})
            state_t: Optimizer state at step t
            batch: Training batch at step t
            
        Returns:
            Gradient w.r.t. metaparameters z (B_t)
        """
        # Reconstruct model at step t
        temp_model = SimpleModel()
        temp_model.load_state_dict(state_t['model_state'])
        temp_model.train()
        
        x, y = batch
        
        # Forward pass with z-dependent loss
        y_pred = temp_model(x)
        loss = self.weighted_loss(y_pred, y, self.z)
        
        # Compute ∂h_t/∂z
        grad_z = torch.autograd.grad(loss, self.z, retain_graph=True)
        
        # Compute B_t = A_t ⋅ ∂h_t/∂z
        B_t = torch.sum(torch.stack([a * g for a, g in zip(A_t, grad_z)]))
        return B_t
    
    def compute_gradient_through_state(self, A_t_plus_1, state_t, batch):
        """Compute gradient through state transition
        
        Args:
            A_t_plus_1: Gradient from step t+1 (∂ϕ/∂s_{t+1})
            state_t: Optimizer state at step t
            batch: Training batch at step t
            
        Returns:
            Gradient w.r.t. state at step t (A_t)
        """
        # Reconstruct model at step t
        temp_model = SimpleModel()
        temp_model.load_state_dict(state_t['model_state'])
        temp_model.train()
        
        x, y = batch
        
        # Forward pass with z-dependent loss
        y_pred = temp_model(x)
        loss = self.weighted_loss(y_pred, y, self.z)
        
        # Compute ∂h_t/∂s_t
        model_params = list(temp_model.parameters())
        grad_s_t = torch.autograd.grad(loss, model_params, create_graph=True)
        
        # Compute A_t = A_{t+1} ⋅ ∂h_t/∂s_t
        A_t = [torch.sum(a * g) for a, g in zip(A_t_plus_1, grad_s_t)]
        return A_t
    
    def step(self, n_steps=10):
        """Perform one step of MGD optimization
        
        Args:
            n_steps: Number of training steps per MGD update
            
        Returns:
            Updated metaparameters z
        """
        # 1. Train with checkpoints
        saved_states = self.train_with_checkpoints(n_steps)
        
        # 2. Compute ∇ϕ(θ_T)
        grad_phi = self.gradient_of_phi_wrt_final_state()
        
        # 3. Backward pass through training
        metagrad = 0
        A_t = grad_phi
        
        # Replay training in reverse
        for i in reversed(range(len(saved_states) - 1)):
            # Get batch at checkpoint i
            batch = next(iter(self.train_loader))
            
            # Compute gradient contributions
            B_t = self.compute_gradient_through_step(A_t, saved_states[i], batch)
            metagrad += B_t
            
            A_t = self.compute_gradient_through_state(A_t, saved_states[i], batch)
        
        # 4. Update metaparameters
        self.z = self.z - self.lr * metagrad
        return self.z

def main():
    """Example usage of the MGD optimizer"""
    # Create synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=10)
    val_loader = DataLoader(val_dataset, batch_size=20)
    
    # Initialize model and metaparameters (data weights)
    model = SimpleModel()
    initial_z = torch.ones(len(train_loader.dataset), requires_grad=True)
    
    # Initialize MGD optimizer
    mgd = MGDOptimizer(model, train_loader, val_loader, initial_z)
    
    # Run MGD optimization
    print("Starting MGD optimization...")
    for epoch in range(5):
        z = mgd.step(n_steps=10)
        print(f"Epoch {epoch+1}, z norm: {torch.norm(z).item():.4f}")
    
    print("Optimization complete!")

if __name__ == "__main__":
    main()
