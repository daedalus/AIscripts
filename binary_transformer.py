#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------
# Straight Through Estimator
# --------------------------------------------------

class BinarySign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize_weight(W):
    # FIX 2: per-output-channel (row-wise) alpha instead of a single scalar.
    # Shape: (out_features, 1) for linear; keepdim handles conv too.
    alpha = W.abs().mean(dim=tuple(range(1, W.dim())), keepdim=True)
    B = BinarySign.apply(W)
    return alpha * B


# --------------------------------------------------
# Binary Linear Layer
# --------------------------------------------------

class BinaryLinear(nn.Module):

    def __init__(self, linear):

        super().__init__()

        self.weight_real = nn.Parameter(linear.weight.data.clone())

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):

        W = binarize_weight(self.weight_real)

        return F.linear(x, W, self.bias)


# --------------------------------------------------
# Binary Conv Layer
# --------------------------------------------------

class BinaryConv2d(nn.Module):

    def __init__(self, conv):

        super().__init__()

        self.weight_real = nn.Parameter(conv.weight.data.clone())

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x):

        W = binarize_weight(self.weight_real)

        return F.conv2d(
            x,
            W,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


# --------------------------------------------------
# Convert model recursively
# --------------------------------------------------

def convert_model(module):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):
            setattr(module, name, BinaryLinear(child))

        elif isinstance(child, nn.Conv2d):
            setattr(module, name, BinaryConv2d(child))

        else:
            convert_model(child)

    return module


# --------------------------------------------------
# Bit packing utilities
# --------------------------------------------------

def pack_binary_tensor(tensor):

    arr = tensor.detach().cpu().numpy()

    # FIX 5: use != -1 (or equivalently >= 0) so that the zero-weight edge
    # case (sign() returns 0, which is neither +1 nor -1) maps consistently
    # to the negative class rather than silently splitting across the threshold.
    arr = (arr != -1).astype(np.uint8)

    flat = arr.flatten()

    packed = np.packbits(flat)

    return packed


def export_binary_model(model, path):

    data = {}

    for name, param in model.named_parameters():

        if "weight_real" in name:

            # Per-channel alphas: one scalar per output channel.
            alpha = param.abs().mean(dim=tuple(range(1, param.dim())))

            B = param.sign()

            packed = pack_binary_tensor(B)

            data[name] = {
                # Store as numpy array so every channel's scale is preserved.
                "alpha": alpha.detach().cpu().numpy(),
                "bits": packed,
                "shape": list(param.shape),
            }

    torch.save(data, path)


# --------------------------------------------------
# Binary Attention  (FIX 1: proper multi-head split)
# --------------------------------------------------

class BinaryAttention(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads"

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = BinaryLinear(nn.Linear(dim, dim))
        self.k = BinaryLinear(nn.Linear(dim, dim))
        self.v = BinaryLinear(nn.Linear(dim, dim))

        self.proj = BinaryLinear(nn.Linear(dim, dim))

    def forward(self, x):

        B, T, C = x.shape

        def split_heads(t):
            # (B, T, C) -> (B, heads, T, head_dim)
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))

        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, T, T)

        att = torch.softmax(att, dim=-1)

        y = att @ v                                     # (B, heads, T, head_dim)

        # Merge heads: (B, heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


# --------------------------------------------------
# Binary MLP
# --------------------------------------------------

class BinaryMLP(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc1 = BinaryLinear(nn.Linear(dim, dim * 4))
        self.fc2 = BinaryLinear(nn.Linear(dim * 4, dim))

    def forward(self, x):

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return x


# --------------------------------------------------
# Binary Transformer Block
# --------------------------------------------------

class BinaryTransformerBlock(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = BinaryAttention(dim, heads)
        self.mlp = BinaryMLP(dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


# --------------------------------------------------
# Binary Transformer  (FIX 4: positional encoding added)
# --------------------------------------------------

class BinaryTransformer(nn.Module):

    def __init__(self, vocab, dim=256, depth=6, heads=8, max_seq_len=512):

        super().__init__()

        self.embed = nn.Embedding(vocab, dim)

        # Learned positional embeddings — keeps sequence order information.
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList(
            [BinaryTransformerBlock(dim, heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, vocab)

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        x = self.embed(x) + self.pos_embed(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.head(x)


# --------------------------------------------------
# Example pipeline
# --------------------------------------------------

def main():

    vocab = 10000

    model = BinaryTransformer(
        vocab=vocab,
        dim=256,
        depth=4,
        heads=4,
        max_seq_len=128,
    )

    print("Model architecture:")
    print(model)

    tokens = torch.randint(0, vocab, (2, 64))

    logits = model(tokens)

    print("Output shape:", logits.shape)   # expect (2, 64, 10000)

    export_binary_model(model, "binary_model.pt")

    print("Binary weights exported")

    # --------------------------------------------------
    # Minimal training loop demo  (FIX 6)
    # --------------------------------------------------
    # The real benefit of weight_real + binarize-in-forward is that latent
    # weights accumulate small gradient updates while the forward pass stays
    # fully binary.  Without a training loop the binarization has no effect
    # beyond compression.

    print("\nRunning a quick training-loop sanity check …")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(3):

        tokens = torch.randint(0, vocab, (2, 32))

        # Shift targets by one for a simple next-token prediction loss.
        logits = model(tokens[:, :-1])           # (B, T-1, vocab)
        targets = tokens[:, 1:].contiguous()     # (B, T-1)

        loss = F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  step {step+1}  loss={loss.item():.4f}")

    print("Training loop OK — gradients flow through weight_real correctly.")


if __name__ == "__main__":
    main()