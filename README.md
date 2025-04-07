# AIscripts #

**AIscripts** is a collection of experimental algorithms, machine learning components, and optimization tools inspired by various domains. This repository serves as a playground for exploring computational efficiency, mathematical models, and AI-driven solutions.

![AIscripts](https://img.shields.io/badge/Status-Experimental-orange)

---

## Features

### Core Components
- **Matrix Operations**:  
  `GEMM_tiled.py` - Tiled implementation of General Matrix Multiply (GEMM).  
  `KMM.py` - Karatsuba Matrix Multiplication utilities.
  `MGD.py` - Meta Gradient Descent optimizer.  

- **AI/ML Tools**:  
  `SeedLM.py` - Seed and coefficient selection algorithm for a single weight block.
  `aicommit.py` - AI-assisted Git commit message generator.  
  `barcode_ssd_mobilenet_v1_dmp25_quant.tflite.runner.py` - TensorFlow Lite runner for SSD MobileNet-based barcode detection.  

- **Probabilistic Data Structures**:  
  `bloomfilter.py` - Classic Bloom filter implementation.  
  `cuckoofilter.py` - Space-efficient Cuckoo filter.  

- **Optimization & Cryptography**:  
  `ZeroMerge.py` - Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs. 
  `minimal_feistel_network.py` - Compact Feistel network cipher.  

- **Combinatorial Algorithms**:  
  `polyomino_tiling.py` - Polyomino tiling solver.  
  `cassowary.py` - Constraint-solving algorithm implementation.  


- **Misc**:
  `calcDa.py` - Density altitude calculator.
  `mdnscan.py` - mdns simple scanner.  
  `mbf.py` - Minimal bloom filter implementation. 

---

## Getting Started

### Prerequisites
- Python 3.8+
- `pip install -r requirements.txt`  
  *(Example dependencies: NumPy, TensorFlow Lite, PyTorch)*

### Basic Usage
```bash
# Run matrix multiplication benchmark
python GEMM_tiled.py --size 1024

# Generate AI-assisted commit message
python aicommit.py --diff <your_git_diff>
