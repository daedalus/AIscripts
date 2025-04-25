"""
70% Size, 100% Accuracy: 
Lossless LLM Compression for Efficient GPU 
Inference via Dynamic-Length Float
"""

import torch
import numpy as np
import heapq
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import copy

# === Huffman coding (basic) ===
def build_huffman_codebook(freqs):
    heap = [[weight, [symbol, ""]] for symbol, weight in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for p in lo[1:]: p[1] = '0' + p[1]
        for p in hi[1:]: p[1] = '1' + p[1]
        heapq.heappush(heap, [lo[0]+hi[0]] + lo[1:] + hi[1:])
    return {symbol: code for _, symbol, code in heap[0][1:]}

class HuffmanDecoder:
    def __init__(self, codebook):
        self.root = {}
        for symbol, code in codebook.items():
            node = self.root
            for bit in code:
                node = node.setdefault(bit, {})
            node['symbol'] = symbol

    def decode_stream(self, bitstream):
        output, node = [], self.root
        for bit in bitstream:
            node = node[bit]
            if 'symbol' in node:
                output.append(node['symbol'])
                node = self.root
        return output

# === Conversion helpers ===
def float32_to_bf16(tensor):
    return (tensor.view(torch.uint32) >> 16).type(torch.uint16)

def bf16_to_float32(bf16):
    return (bf16.type(torch.uint32) << 16).view(torch.float32)

# === DFloat11 simulation ===
def compress_decompress_tensor(tensor):
    flat = tensor.contiguous().view(torch.float32)
    bf16 = float32_to_bf16(flat)

    signs = (bf16 >> 15) & 1
    exps = (bf16 >> 7) & 0xFF
    mans = bf16 & 0x7F

    exp_freqs = Counter(exps.tolist())
    codebook = build_huffman_codebook(exp_freqs)
    decoder = HuffmanDecoder(codebook)

    encoded_exp_bits = ''.join(codebook[e] for e in exps.tolist())
    decoded_exps = decoder.decode_stream(encoded_exp_bits)
    decoded_exps = torch.tensor(decoded_exps, dtype=torch.uint16)

    bf16_rec = (signs << 15) | (decoded_exps << 7) | mans
    return bf16_to_float32(bf16_rec).view_as(tensor)

# === Load model and tokenizer ===
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
orig_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# === Compress and reconstruct model ===
dfloat_model = copy.deepcopy(orig_model)
for name, param in dfloat_model.named_parameters():
    if param.requires_grad and param.dim() > 1:
        param.data = compress_decompress_tensor(param.data)

# === Inference comparison ===
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    orig_logits = orig_model(**inputs).logits
    dfloat_logits = dfloat_model(**inputs).logits

orig_top1 = torch.argmax(orig_logits[0, -1])
dfloat_top1 = torch.argmax(dfloat_logits[0, -1])

print("Original prediction:", tokenizer.decode([orig_top1.item()]))
print("DFloat11 prediction:", tokenizer.decode([dfloat_top1.item()]))
print("Max logit diff:", torch.max(torch.abs(orig_logits - dfloat_logits)).item())
