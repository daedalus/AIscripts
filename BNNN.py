"""
arXiv:2509.07025v1 [cs.LG] 07 Sep 2025
1 BIT IS ALL WE NEED: Binary Normalized Neural Networks
"""

# binary_normalized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# -------------------------
# Helpers
# -------------------------
def layer_mean(t: torch.Tensor) -> torch.Tensor:
    """Return scalar mean for a parameter tensor (same dtype/device)."""
    return t.mean()

def quantize_to_01(t: torch.Tensor) -> torch.Tensor:
    """
    Quantize to {0,1} using threshold equal to the mean of t (scalar).
    Paper: pb = 1 if p > pmean else 0
    """
    # compute scalar mean
    m = layer_mean(t)
    return (t > m).to(dtype=t.dtype)

class QuantizeSTE(torch.autograd.Function):
    """
    Straight-through estimator quantization function:
      forward: returns binary tensor {0,1} using layer mean threshold
      backward: pass gradient unchanged (identity)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return quantize_to_01(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through: propagate gradient as-is to full precision params
        return grad_output


def normalize_per_sample(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Normalize each example (sample) to zero mean and unit std.
    For linear outputs: x shape = (batch, features) -> normalize over features dim=-1
    For conv outputs: x shape = (batch, C, H, W) -> normalize over (C,H,W) per sample.
    """
    if x.dim() == 2:
        # (B, F)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std
    elif x.dim() == 4:
        # (B, C, H, W) -> compute per-sample mean/std across (C,H,W)
        b = x.shape[0]
        # flatten per sample
        flattened = x.view(b, -1)
        mean = flattened.mean(dim=-1, keepdim=True)
        std = flattened.std(dim=-1, keepdim=True) + eps
        normalized = (flattened - mean) / std
        return normalized.view_as(x)
    elif x.dim() == 3:
        # (B, T, F) common for transformer token features -> normalize over last dim F
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std
    else:
        # fallback: normalize across last dim
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std


# -------------------------
# Binary normalized linear (BNFCL)
# -------------------------
class BinaryNormalizedLinear(nn.Module):
    """
    Binary Normalized Fully Connected Layer (BNFCL).
    Keeps full-precision params (self.weight_fp / self.bias_fp).
    Forward uses binarized version computed by QuantizeSTE.
    """
    def __init__(self, in_features: int, out_features: int, activation: Optional[str] = None):
        super().__init__()
        # store parameters in same naming as typical nn.Linear but full-precision
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activation
        nn.init.xavier_uniform_(self.weight)

    def _get_binarized(self, train_mode: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if train_mode:
            # Wq = W + NoGrad(Quant(W) - W)  -> implemented by adding detached difference
            Wq = self.weight + (QuantizeSTE.apply(self.weight) - self.weight).detach()
            bq = self.bias + (QuantizeSTE.apply(self.bias) - self.bias).detach()
        else:
            Wq = QuantizeSTE.apply(self.weight)
            bq = QuantizeSTE.apply(self.bias)
        return Wq, bq

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> torch.Tensor:
        """
        x: shape (..., in_features) where last dim is in_features
        train_mode: if True follow Algorithm 1 (training forward); else quantize directly for inference
        """
        Wq, bq = self._get_binarized(train_mode)
        # use F.linear (handles batch and extra dims)
        z = F.linear(x, Wq, bq)
        z = normalize_per_sample(z)
        a = z
        if self.activation is None:
            return a
        act = self.activation.lower()
        if act == "relu":
            return F.relu(a)
        if act == "gelu":
            return F.gelu(a)
        if act == "softmax":
            # softmax across last dim (classes)
            return F.softmax(a, dim=-1)
        if act == "tanh":
            return torch.tanh(a)
        return a

    def to_binarized_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return binarized (0/1) weights and biases using mean threshold."""
        return quantize_to_01(self.weight), quantize_to_01(self.bias)


# -------------------------
# Binary normalized Conv2d (BNCVL)
# -------------------------
class BinaryNormalizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, activation: Optional[str] = None):
        super().__init__()
        # weight shape: (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.activation = activation
        nn.init.xavier_uniform_(self.weight)

    def _get_binarized(self, train_mode: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if train_mode:
            Wq = self.weight + (QuantizeSTE.apply(self.weight) - self.weight).detach()
            bq = self.bias + (QuantizeSTE.apply(self.bias) - self.bias).detach()
        else:
            Wq = QuantizeSTE.apply(self.weight)
            bq = QuantizeSTE.apply(self.bias)
        return Wq, bq

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> torch.Tensor:
        # x: (B, C_in, H, W)
        Wq, bq = self._get_binarized(train_mode)
        z = F.conv2d(x, Wq, bias=bq, stride=self.stride, padding=self.padding)
        z = normalize_per_sample(z)
        if self.activation is None:
            return z
        if self.activation.lower() == "relu":
            return F.relu(z)
        return z

    def to_binarized_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return quantize_to_01(self.weight), quantize_to_01(self.bias)


# -------------------------
# Binary Embedding (BEMBL)
# -------------------------
class BinaryEmbedding(nn.Module):
    """
    Implements Algorithm 4: token and positional embeddings built using BNFCL linear layers.
    Behavior:
      - Input seq tensor: LongTensor of shape (batch, seq_len) with token ids in [0, vocab_size-1]
      - Produces embeddings shape (batch, seq_len, emb_dim)
    Internals: token embedding and position embedding both use BinaryNormalizedLinear applied to one-hot vectors.
    To avoid constructing huge one-hot matrices on large vocabs, we do a sparse approach: index-select from a learnable
    full-precision embedding matrix and then quantize that matrix for forward. But to remain faithful to the paper's
    algorithm (they used BNFCL on one-hot), here we create a lightweight internal full-precision lookup that is quantized.
    """
    def __init__(self, max_len: int, emb_dim: int, vocab_size: int):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # We'll implement token embedding as a weight matrix (vocab_size x emb_dim) stored as full-precision param
        # and then use quantization in forward similarly to BNFCL. Same for position embeddings.
        self.token_table = nn.Parameter(torch.empty(vocab_size, emb_dim))
        self.pos_table = nn.Parameter(torch.empty(max_len, emb_dim))
        nn.init.xavier_uniform_(self.token_table)
        nn.init.xavier_uniform_(self.pos_table)

    def forward(self, seq: torch.LongTensor, train_mode: bool = True) -> torch.Tensor:
        """
        seq: (B, T) long tensor with tokens [0..vocab_size-1]
        returns: (B, T, emb_dim)
        """
        # quantize token and pos tables
        if train_mode:
            token_table_q = self.token_table + (QuantizeSTE.apply(self.token_table) - self.token_table).detach()
            pos_table_q = self.pos_table + (QuantizeSTE.apply(self.pos_table) - self.pos_table).detach()
        else:
            token_table_q = QuantizeSTE.apply(self.token_table)
            pos_table_q = QuantizeSTE.apply(self.pos_table)

        # token embeddings via index select
        tk_emb = F.embedding(seq, token_table_q)  # (B, T, emb_dim)
        # position indices
        device = seq.device
        seq_len = seq.shape[1]
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq.shape[0], -1)  # (B, T)
        pos_emb = F.embedding(pos_idx, pos_table_q)  # (B, T, emb_dim)

        tk_pos_emb = tk_emb + pos_emb
        return tk_pos_emb


# -------------------------
# Binary Multi-Head Attention (BATL)
# -------------------------
class BinaryMultiHeadAttention(nn.Module):
    """
    Binary Multi-Head Attention following Algorithm 6.
    Uses BinaryNormalizedLinear for projections Q,K,V and final projection.
    All linear projections return shape (B, T, emb_dim).
    """
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.key_dim = emb_dim // num_heads

        # projection layers are binary normalized linear layers
        self.q_proj = BinaryNormalizedLinear(emb_dim, emb_dim, activation=None)
        self.k_proj = BinaryNormalizedLinear(emb_dim, emb_dim, activation=None)
        self.v_proj = BinaryNormalizedLinear(emb_dim, emb_dim, activation=None)
        self.out_proj = BinaryNormalizedLinear(emb_dim, emb_dim, activation=None)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, train_mode: bool = True) -> torch.Tensor:
        """
        query/key/value: (B, T, emb_dim)
        mask: optional (B, T, T) boolean mask (0 allowed positions, 1 masked?) Paper: Where(mask==0, -1e10, scale_dot)
              We'll assume mask is a causal mask where valid positions are 1 and invalid are 0. If mask is None, no mask.
        Returns projection: (B, T, emb_dim)
        """
        B, T, _ = query.shape
        num_heads = self.num_heads
        key_dim = self.key_dim

        # linear projections (binary normalized)
        Q = self.q_proj(query, train_mode=train_mode)  # (B, T, emb_dim)
        K = self.k_proj(key, train_mode=train_mode)
        V = self.v_proj(value, train_mode=train_mode)

        # reshape to (B, num_heads, T, key_dim)
        def split_heads(x):
            return x.view(B, T, num_heads, key_dim).permute(0, 2, 1, 3)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # scaled dot product
        # Qh: (B, H, T, d), Kh: (B, H, T, d) -> scores (B, H, T, T)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (key_dim ** 0.5)

        if mask is not None:
            # mask shape must be broadcastable to (B, H, T, T)
            # Paper: scale_dot = Where(mask == 0, -1.0e-10, scale_dot)
            # We'll use a large negative for masked positions to zero them after softmax:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T, T)
            scores = torch.where(mask_expanded == 0, torch.tensor(-1e10, device=scores.device, dtype=scores.dtype), scores)

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        A = torch.matmul(attn, Vh)  # (B, H, T, d)

        # merge heads back
        A = A.permute(0, 2, 1, 3).contiguous().view(B, T, num_heads * key_dim)  # (B, T, emb_dim)

        projection = self.out_proj(A, train_mode=train_mode)  # final linear proj
        return projection


# -------------------------
# Binary Transformer Block (BTFB)
# -------------------------
class BinaryTransformerBlock(nn.Module):
    """
    Binary transformer block implementing Algorithm 5.
    Structure:
      attention_output = BATL(...)(seq, seq, seq)
      add & normalize
      ff = BNFCL(units=ff_dim, activation='gelu')(add_norm)
      ff = BNFCL(units=emb_dim)(ff)
      add & normalize -> output
    """
    def __init__(self, emb_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attention = BinaryMultiHeadAttention(emb_dim, num_heads)
        self.norm1 = lambda x: normalize_per_sample(x)  # paper uses simple normalize (no learnable params)
        self.ffn1 = BinaryNormalizedLinear(emb_dim, ff_dim, activation="gelu")
        self.ffn2 = BinaryNormalizedLinear(ff_dim, emb_dim, activation=None)
        self.norm2 = lambda x: normalize_per_sample(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, train_mode: bool = True) -> torch.Tensor:
        attn_out = self.attention(x, x, x, mask=mask, train_mode=train_mode)
        add_norm = self.norm1(x + attn_out)
        ffn_out = self.ffn1(add_norm, train_mode=train_mode)
        ffn_out = self.ffn2(ffn_out, train_mode=train_mode)
        output = self.norm2(add_norm + ffn_out)
        return output


# -------------------------
# Binary Convolutional Model (BCVNN) - Algorithm 7
# -------------------------
class BinaryConvModel(nn.Module):
    """
    Implements Algorithm 7 (BCVNN). Expects input images shaped (B, 3, H, W).
    filter_size f is kernel dimension used for all conv layers.
    """
    def __init__(self, filter_size: int = 3, num_classes: int = 101):
        super().__init__()
        f = filter_size
        # first block
        self.c1a = BinaryNormalizedConv2d(3, 32, kernel_size=f, stride=1, padding=f//2, activation="relu")
        self.c1b = BinaryNormalizedConv2d(32, 32, kernel_size=f, stride=1, padding=f//2, activation="relu")
        # second block
        self.c2a = BinaryNormalizedConv2d(32, 64, kernel_size=f, stride=1, padding=f//2, activation="relu")
        self.c2b = BinaryNormalizedConv2d(64, 64, kernel_size=f, stride=1, padding=f//2, activation="relu")
        # third block
        self.c3a = BinaryNormalizedConv2d(64, 64, kernel_size=f, stride=1, padding=f//2, activation="relu")
        self.c3b = BinaryNormalizedConv2d(64, 64, kernel_size=f, stride=1, padding=f//2, activation="relu")
        # fourth block
        self.c4a = BinaryNormalizedConv2d(64, 128, kernel_size=f, stride=1, padding=f//2, activation="relu")
        self.c4b = BinaryNormalizedConv2d(128, 128, kernel_size=f, stride=1, padding=f//2, activation="relu")
        # fifth block
        self.c5a = BinaryNormalizedConv2d(128, 256, kernel_size=f, stride=1, padding=f//2, activation="relu")
        self.c5b = BinaryNormalizedConv2d(256, 256, kernel_size=f, stride=1, padding=f//2, activation="relu")

        # FC layers
        self.fc1 = BinaryNormalizedLinear(256, 256, activation="relu")  # note: after global avg we have (B, 256)
        self.fc2 = BinaryNormalizedLinear(256, 256, activation="relu")
        self.fc_out = BinaryNormalizedLinear(256, num_classes, activation="softmax")

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> torch.Tensor:
        # x: (B, 3, H, W), assume H,W large enough for pooling operations
        def mp(t):
            return F.max_pool2d(t, kernel_size=2, stride=2)

        x = self.c1a(x, train_mode=train_mode)
        x = self.c1b(x, train_mode=train_mode)
        x = mp(x)

        x = self.c2a(x, train_mode=train_mode)
        x = self.c2b(x, train_mode=train_mode)
        x = mp(x)

        x = self.c3a(x, train_mode=train_mode)
        x = self.c3b(x, train_mode=train_mode)
        x = mp(x)

        x = self.c4a(x, train_mode=train_mode)
        x = self.c4b(x, train_mode=train_mode)
        x = mp(x)

        x = self.c5a(x, train_mode=train_mode)
        x = self.c5b(x, train_mode=train_mode)
        # global average pool -> (B, C, 1, 1) -> view (B, C)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(x.size(0), -1)

        x = self.fc1(x, train_mode=train_mode)
        x = self.fc2(x, train_mode=train_mode)
        out = self.fc_out(x, train_mode=train_mode)
        return out

    def to_binarized_state(self) -> dict:
        """Return a state dict where weights/biases are binarized (0/1)."""
        state = {}
        for name, param in self.named_parameters():
            # quantize every parameter tensor by layer. For simplicity treat each param independently.
            state[name] = quantize_to_01(param).detach().cpu()
        return state


# -------------------------
# Binary Language Decoder (BLM) - Algorithm 8
# -------------------------
class BinaryLanguageDecoder(nn.Module):
    def __init__(self, max_len: int, emb_dim: int, num_heads: int, num_blocks: int,
                 vocab_size: int, mlp_units0: int, mlp_units1: int):
        super().__init__()
        self.embedding = BinaryEmbedding(max_len, emb_dim, vocab_size)
        self.norm_embed = lambda x: normalize_per_sample(x)
        self.blocks = nn.ModuleList([BinaryTransformerBlock(emb_dim, num_heads, ff_dim=2*emb_dim) for _ in range(num_blocks)])
        # MLP head as BNFCL: note the paper applies these to the transformer output per-token
        self.mlp0 = BinaryNormalizedLinear(emb_dim, mlp_units0, activation="gelu")
        self.mlp1 = BinaryNormalizedLinear(mlp_units0, mlp_units1, activation="gelu") if mlp_units1 > 0 else None
        self.out = BinaryNormalizedLinear(mlp_units1 if mlp_units1>0 else mlp_units0, vocab_size, activation="softmax")

    def forward(self, seq: torch.LongTensor, mask: Optional[torch.Tensor] = None, train_mode: bool = True) -> torch.Tensor:
        """
        seq: (B, T) long tensor
        mask: optional (B, T, T) mask with 1 for allowed positions and 0 for masked (paper uses Where(mask==0, -1e-10,...))
        returns: (B, T, vocab_size)
        """
        x = self.embedding(seq, train_mode=train_mode)
        x = self.norm_embed(x)
        for block in self.blocks:
            x = block(x, mask=mask, train_mode=train_mode)

        # apply MLP head token-wise (x is (B, T, emb_dim))
        features = self.mlp0(x, train_mode=train_mode)  # (B, T, mlp_units0)
        if self.mlp1 is not None:
            features = self.mlp1(features, train_mode=train_mode)  # (B, T, mlp_units1)
        probs = self.out(features, train_mode=train_mode)  # (B, T, vocab_size)
        return probs

    def to_binarized_state(self) -> dict:
        state = {}
        for name, param in self.named_parameters():
            state[name] = quantize_to_01(param).detach().cpu()
        return state


# -------------------------
# Example usage / quick smoke tests
# -------------------------
if __name__ == "__main__":
    # Smoke test BCVNN
    device = torch.device("cpu")
    model_img = BinaryConvModel(filter_size=3, num_classes=101).to(device)
    dummy_image = torch.randn(2, 3, 256, 256, device=device)  # batch 2
    out_img = model_img(dummy_image, train_mode=True)
    print("BCVNN output shape (train_mode):", out_img.shape)

    # Convert to binarized state
    bin_state = model_img.to_binarized_state()
    print("Sample binarized param keys:", list(bin_state.keys())[:6])

    # Smoke test BLM
    max_len = 16
    vocab_size = 30522
    emb_dim = 128  # keep small for smoke test
    num_heads = 8
    num_blocks = 2
    mlp0, mlp1 = 256, 128

    model_lang = BinaryLanguageDecoder(max_len=max_len, emb_dim=emb_dim, num_heads=num_heads,
                                       num_blocks=num_blocks, vocab_size=vocab_size, mlp_units0=mlp0,
                                       mlp_units1=mlp1).to(device)

    dummy_seq = torch.randint(low=0, high=vocab_size, size=(2, max_len), dtype=torch.long, device=device)
    # Create causal mask (1 allowed, 0 masked) typical triangular mask
    seq_len = dummy_seq.shape[1]
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)  # upper triangle 1
    mask = (causal == 0).unsqueeze(0).expand(2, -1, -1).to(dtype=torch.uint8)  # (B, T, T) with 1 allowed, 0 masked

    out_lang = model_lang(dummy_seq, mask=mask, train_mode=True)
    print("BLM output shape (train_mode):", out_lang.shape)
