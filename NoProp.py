"""
NoProp: Training Neural Networks without Back-propagation or Forward-propagation
Qinyu Li, Yee Whye Teh, Razvan Pascanu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Hyperparameters
T = 10  # Number of diffusion steps
embed_dim = 20  # Label embedding dimension
input_channels = 1  # MNIST: 1, CIFAR: 3
hidden_dim = 128
batch_size = 128
lr = 1e-3
epochs = 100

# Diffusion noise schedule
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Network components
class ResidualBlock(nn.Module):
    """Diffusion dynamics block u_t(z_{t-1}, x)"""
    def __init__(self, embed_dim, input_channels):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, hidden_dim)  # For MNIST 28x28
        )
        
        self.label_processor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, z, x):
        x_emb = self.image_encoder(x)
        z_emb = self.label_processor(z)
        combined = torch.cat([x_emb, z_emb], dim=-1)
        return self.combined(combined)

class NoProp(nn.Module):
    def __init__(self, num_classes, embed_dim, T):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        
        # Learnable label embeddings
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        # Create T residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(embed_dim, input_channels) for _ in range(T)]
        )
        
        # Final classification layer
        self.fc_out = nn.Linear(embed_dim, num_classes)
        
        # Initialize noise schedule
        betas = cosine_beta_schedule(T)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)

    def forward_process(self, y, t):
        """Add noise to label embeddings"""
        emb = self.label_embed(y)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[t])
        noise = torch.randn_like(emb)
        return sqrt_alpha * emb + sqrt_one_minus_alpha * noise, noise

    def train_step(self, x, y, t):
        z_t, noise = self.forward_process(y, t)
        pred_noise = self.blocks[t](z_t.detach(), x)
        
        # L2 loss for denoising
        loss = F.mse_loss(pred_noise, noise)
        
        # Classification loss
        logits = self.fc_out(z_t)
        cls_loss = F.cross_entropy(logits, y)
        
        return loss + cls_loss

    def inference(self, x):
        """Generate predictions through iterative denoising"""
        # Start with Gaussian noise
        z = torch.randn(x.size(0), embed_dim).to(x.device)
        
        for t in reversed(range(self.T)):
            # Predict denoised embedding
            pred = self.blocks[t](z, x)
            
            # Update z using reverse process
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else 1.0
            beta_t = 1 - alpha_t / alpha_t_prev
            
            z = (z - beta_t * pred) / torch.sqrt(1 - beta_t)
            z += torch.sqrt(beta_t) * torch.randn_like(z)
        
        # Final classification
        return self.fc_out(z)

# Training loop
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Sample random timestep for each batch
        t = torch.randint(0, T, (x.size(0),), device=device)
        
        # Train each block sequentially
        for t_step in range(T):
            mask = (t == t_step)
            if mask.sum() == 0:
                continue
                
            x_t = x[mask]
            y_t = y[mask]
            
            optimizer.zero_grad()
            loss = model.train_step(x_t, y_t, t_step)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation
def test(model, test_loader):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model.inference(x)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
    
    return correct / len(test_loader.dataset)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoProp(num_classes=10, embed_dim=embed_dim, T=T).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
for epoch in range(1, epochs+1):
    train_loss = train(model, train_loader, optimizer)
    test_acc = test(model, test_loader)
    print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")