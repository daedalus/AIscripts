"""
SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators
Rasoul Shafipour, David Harrison, Maxwell Horton, Jeffrey Marker, Houman Bedayat, Sachin Mehta, Mohammad Rastegari, Mahyar Najibi, Saman Naderiparizi
"""

import numpy as np

def lfsr_next(state, K, taps):
    """
    Compute the next state of an LFSR.
    
    state: current state (integer with K bits)
    K: length of the register
    taps: list of tap indices (with bit index 0 as LSB)
    """
    # XOR the tapped bits
    xor_val = 0
    for tap in taps:
        # Extract bit at position tap
        xor_val ^= (state >> tap) & 1
    # Shift right by one and set new MSB with xor_val
    new_state = (xor_val << (K - 1)) | (state >> 1)
    # Ensure new_state is K-bit (mask if needed)
    new_state &= (1 << K) - 1
    return new_state

def generate_V(seed, C, P, K, taps):
    """
    Generate the V(s) matrix using an LFSR initialized with the given seed.
    
    Parameters:
        seed: initial seed (integer, nonzero and < 2^K)
        C: block size (number of rows)
        P: latent dimension (number of columns)
        K: LFSR length (in bits)
        taps: list of tap positions for the LFSR
        
    Returns:
        V: a (C x P) matrix of integers from the LFSR sequence.
           The matrix is filled row-wise using the numbers generated
           starting from the first state after the seed.
    """
    total = C * P
    V = np.empty(total, dtype=np.int32)
    state = seed
    # Generate total numbers (do not use the seed itself, but the next states)
    for i in range(total):
        state = lfsr_next(state, K, taps)
        V[i] = state
    return V.reshape(C, P)

def compute_U(V, K):
    """
    Normalize V(s) to obtain U(s) with values in [-1, 1].
    
    Using the formula:
      U = (V - 2^(K-1)) / (2^(K-1) - 1)
    """
    midpoint = 1 << (K - 1)  # 2^(K-1)
    denom = midpoint - 1
    U = (V.astype(np.float32) - midpoint) / denom
    return U

def quantize_coeffs(t):
    """
    Quantize the coefficient vector t to 4-bit two's complement with a shared exponent.
    
    For each element, we compute the shared exponent e = max(floor(log2(|t_i|))) over nonzero t.
    Then, each coefficient is scaled by 2^e, rounded, and clipped to the range [-8, 7].
    
    Returns:
        t_quant: the quantized coefficient vector (reconstructed back to float)
        e: the shared exponent (an integer)
    """
    t = np.array(t, dtype=np.float32)
    abs_nonzero = np.abs(t[np.nonzero(t)])
    if abs_nonzero.size == 0:
        e = 0
    else:
        e = int(np.floor(np.log2(np.max(abs_nonzero)))) if np.max(abs_nonzero) > 0 else 0
    scale = 2 ** e
    # Scale and round to nearest integer
    q = np.round(t / scale).astype(np.int32)
    # Clip to 4-bit two's complement range: [-8, 7]
    q = np.clip(q, -8, 7)
    # Reconstruct quantized coefficients
    t_quant = q.astype(np.float32) * scale
    return t_quant, e

def seedlm_select(w, C, P, K, taps):
    """
    Implements Algorithm 1: Seed and Coefficient Selection for a weight block.
    
    Inputs:
      w: a weight block vector of shape (C,)
      C: block size (number of elements)
      P: latent dimension (number of coefficients)
      K: LFSR bit-length (number of bits in the seed)
      taps: list of tap positions for the LFSR (depends on K)
    
    Returns:
      best_seed: the seed s that minimizes reconstruction error
      best_t: the quantized coefficient vector corresponding to best_seed
      best_e: the shared exponent used for quantization
      best_error: the reconstruction error achieved
    """
    best_seed = None
    best_t = None
    best_e = None
    best_error = np.inf

    # Total number of candidate seeds is 2^K - 1.
    # For demonstration purposes, you might want to loop over a smaller set.
    N = (1 << K) - 1

    # Loop over all candidate seeds (note: for large K this loop can be very slow)
    for seed in range(1, N + 1):
        # Generate V(s) and compute U(s)
        V = generate_V(seed, C, P, K, taps)
        U = compute_U(V, K)
        # Compute pseudo-inverse of U (using np.linalg.pinv)
        U_dagger = np.linalg.pinv(U)
        # Project w onto the subspace: t = U_dagger * w
        t = U_dagger.dot(w)
        # Quantize t to obtain 4-bit coefficients and a shared exponent
        t_quant, e = quantize_coeffs(t)
        # Compute reconstruction error
        error = np.linalg.norm(w - U.dot(t_quant))
        if error < best_error:
            best_error = error
            best_seed = seed
            best_t = t_quant
            best_e = e

    return best_seed, best_t, best_e, best_error

# Example usage:
if __name__ == '__main__':
    # Set hyperparameters (for demonstration we choose small numbers)
    # For a 4-bit compression configuration as in the paper (Table 1), one choice is:
    #   M = 4 bits per element, C = 8, P = 3, K = 16.
    # For a quick demo, we use K = 4 so the candidate space is small.
    C = 8      # block size
    P = 3      # latent dimension
    K = 16      # LFSR bit-length; for full-scale demo use K=16
    # Define taps for K=4. (For K=4 in the paper the default taps might be [0,1] but here we choose a simple one.)
    taps_dict = {
        4: [0, 1],
        16: [0, 1, 3, 12]  # as given in the paper for K=16
    }
    taps = taps_dict.get(K)
    if taps is None:
        raise ValueError("Taps for K={} are not defined.".format(K))
    
    # Create a sample weight block vector w (for example, from a pretrained model)
    # Here we create a random vector (in practice, w comes from the model weights)
    np.random.seed(42)
    w = np.random.randn(C).astype(np.float32)

    best_seed, best_t, best_e, best_error = seedlm_select(w, C, P, K, taps)
    
    print("Best seed:", best_seed)
    print("Best quantized coefficients (t):", best_t)
    print("Shared exponent (e):", best_e)
    print("Reconstruction error:", best_error)

