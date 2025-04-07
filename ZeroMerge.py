"""
ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs
Xin Liu, Pei Liu, Guoming Tang
https://arxiv.org/abs/2503.10714
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroMergeCache:
    def __init__(self, total_budget, head_dim, alpha=0.6, decay=0.98):
        self.B_c = int(total_budget * 0.44)  # Context
        self.B_r = int(total_budget * 0.22)  # Residual
        self.B_p = total_budget - self.B_c - self.B_r  # Proximity
        
        self.alpha = alpha
        self.decay = decay
        self.head_dim = head_dim
        
        # Cache storage
        self.context = {'keys': [], 'values': [], 'scores': []}
        self.residual = {'keys': [], 'values': [], 'counts': []}
        self.proximity = {'keys': [], 'values': []}

    def _update_scores(self, attention_weights):
        """Update scores with geometric decay"""
        self.context['scores'] = [s * self.decay + a for s, a in 
                                zip(self.context['scores'], attention_weights)]

    def _merge_to_residual(self, keys, values):
        """Merge tokens into residual slots using similarity"""
        for k, v in zip(keys, values):
            if len(self.residual['keys']) < self.B_r:
                self.residual['keys'].append(k)
                self.residual['values'].append(v)
                self.residual['counts'].append(1)
                continue

            # Find closest residual slot
            similarities = [F.cosine_similarity(k, rk, dim=0) 
                          for rk in self.residual['keys']]
            r_idx = torch.argmax(torch.stack(similarities))

            # Momentum update
            count = self.residual['counts'][r_idx]
            self.residual['keys'][r_idx] = (count * self.residual['keys'][r_idx] + k) / (count + 1)
            self.residual['values'][r_idx] = (count * self.residual['values'][r_idx] + v) / (count + 1)
            self.residual['counts'][r_idx] += 1

    def update_cache(self, new_keys, new_values, attention_weights):
        """Update cache with new tokens and attention weights"""
        # Update existing scores
        if self.context['scores']:
            self._update_scores(attention_weights[:len(self.context['scores'])])
        
        # Add new tokens with initial scores (current attention weights)
        new_scores = attention_weights[len(self.context['scores']):]
        all_keys = self.context['keys'] + new_keys
        all_values = self.context['values'] + new_values
        all_scores = self.context['scores'] + new_scores
        
        # Select top-B_c tokens by score
        sorted_idx = torch.argsort(torch.tensor(all_scores), descending=True)
        selected_idx = sorted_idx[:self.B_c]
        
        # Update context cache
        self.context = {
            'keys': [all_keys[i] for i in selected_idx],
            'values': [all_values[i] for i in selected_idx],
            'scores': [all_scores[i] for i in selected_idx]
        }
        
        # Update proximity with latest tokens
        self.proximity['keys'] = new_keys[-self.B_p:]
        self.proximity['values'] = new_values[-self.B_p:]
        
        # Merge remaining tokens to residual
        remaining_keys = [k for i, k in enumerate(all_keys) if i not in selected_idx]
        remaining_values = [v for i, v in enumerate(all_values) if i not in selected_idx]
        self._merge_to_residual(remaining_keys, remaining_values)

    def compute_attention(self, query):
        """Compute compensated attention output"""
        keys = self.context['keys'] + self.residual['keys'] + self.proximity['keys']
        values = self.context['values'] + self.residual['values'] + self.proximity['values']
        counts = [1]*len(self.context['keys']) + self.residual['counts'] + [1]*len(self.proximity['keys'])
        
        if not keys:
            return torch.zeros_like(query)
            
        # Compute attention scores with compensation
        keys_tensor = torch.stack(keys)
        scores = torch.matmul(query, keys_tensor.T) / (self.head_dim ** 0.5)
        counts_tensor = torch.tensor(counts, device=query.device)
        scores += self.alpha * torch.log(counts_tensor)
        
        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, torch.stack(values))

class ZeroMergeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, cache_budget):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Create cache for each head
        self.caches = [ZeroMergeCache(cache_budget, self.head_dim) 
                      for _ in range(num_heads)]

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        outputs = []
        for t in range(seq_len):
            time_step_output = []
            for h in range(self.num_heads):
                # Get current query
                query = q[:, t, h, :].squeeze(0)
                
                # Compute attention with current cache
                cache = self.caches[h]
                if t == 0:  # Initialize cache
                    cache.update_cache(
                        [k[:, t, h, :].squeeze(0)], 
                        [v[:, t, h, :].squeeze(0)], 
                        torch.tensor([1.0]))
                    continue
                
                # Compute attention weights
                keys = torch.stack(cache.context['keys'] + cache.residual['keys'] + cache.proximity['keys'])
                attn_weights = F.softmax(
                    torch.matmul(query, keys.T) / (self.head_dim ** 0.5),
                    dim=-1
                )
                
                # Update cache with new token
                new_key = k[:, t, h, :].squeeze(0)
                new_value = v[:, t, h, :].squeeze(0)
                cache.update_cache([new_key], [new_value], attn_weights.detach())
                
                # Compute attention output
                attn_output = cache.compute_attention(query)
                time_step_output.append(attn_output)
            
            # Combine head outputs
            outputs.append(torch.stack(time_step_output))
        
        # Format output
        output = torch.stack(outputs).permute(1, 0, 2)
        return self.out_proj(output)

# Example Usage
config = {
    'embed_dim': 512,
    'num_heads': 8,
    'cache_budget': 32,  # Tokens per head
}

model = ZeroMergeAttention(**config)
x = torch.randn(1, 10, config['embed_dim'])  # (batch, seq_len, embed_dim)
output = model(x)
print(f"Output shape: {output.shape}")
