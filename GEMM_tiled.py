"""
https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
"""

import math
import random
import time

def gemm_optimized(A, B, tile=16):
    """
    Multiply matrix A (M×K) by matrix B (K×N) using a tiled algorithm.
    A and B are assumed to be lists of lists (pure Python matrices).
    Returns the resulting matrix C (M×N).
    """
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    
    # Initialize result matrix C with zeros.
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    
    # Determine if math.fma is available (Python 3.9+)
    use_fma = hasattr(math, 'fma')
    
    # Process the matrices in tiles.
    for i in range(0, M, tile):
        for j in range(0, N, tile):
            for k in range(0, K, tile):
                # For each tile, iterate over the submatrix elements.
                for i_inner in range(i, min(i + tile, M)):
                    for j_inner in range(j, min(j + tile, N)):
                        # Start with the current accumulated value.
                        acc = C[i_inner][j_inner]
                        # Process the inner tile.
                        for k_inner in range(k, min(k + tile, K)):
                            a_val = A[i_inner][k_inner]
                            b_val = B[k_inner][j_inner]
                            # Use fma if available for a fused multiply-add.
                            if use_fma:
                                acc = math.fma(a_val, b_val, acc)
                            else:
                                acc += a_val * b_val
                        # Store the updated result.
                        C[i_inner][j_inner] = acc
    return C

# Helper functions to create random matrices and verify the result.
def create_random_matrix(rows, cols):
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def naive_matmul(A, B):
    M, K = len(A), len(A[0])
    N = len(B[0])
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrices_are_close(C1, C2, tol=1e-6):
    M = len(C1)
    N = len(C1[0])
    for i in range(M):
        for j in range(N):
            if abs(C1[i][j] - C2[i][j]) > tol:
                return False
    return True

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark")
    parser.add_argument("--size", type=int, default=64, help="Size of square matrices (M=K=N)")
    parser.add_argument("--tile", type=int, default=16, help="Tile size for optimized GEMM")
    args = parser.parse_args()

    M = K = N = args.size

    A = create_random_matrix(M, K)
    B = create_random_matrix(K, N)

    start_time = time.time()
    C_opt = gemm_optimized(A, B, tile=args.tile)
    elapsed_opt = time.time() - start_time

    start_time = time.time()
    C_naive = naive_matmul(A, B)
    elapsed_naive = time.time() - start_time

    if matrices_are_close(C_opt, C_naive):
        print("The tiled multiplication result matches the naive implementation.")
    else:
        print("Results differ!")

    print(f"Optimized (tiled) implementation time: {elapsed_opt:.6f} seconds")
    print(f"Naive implementation time: {elapsed_naive:.6f} seconds")
