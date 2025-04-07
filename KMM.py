import numpy as np

def kmm_matrix(A, B, w, threshold=8):
    """
    Karatsuba Matrix Multiplication (KMM) for integer matrices.
    
    Each element in A and B is assumed to be a nonnegative integer
    represented in w bits. The algorithm splits each element into a high 
    and low part (using a bit-split at l bits, with l = floor(w/2)) and 
    then recursively computes the product.
    
    When w <= threshold, conventional matrix multiplication (using np.dot)
    is used.
    
    Parameters:
      A, B: 2D numpy arrays of integer type.
      w: bitwidth of each element in A and B.
      threshold: when w <= threshold, use standard multiplication.
    
    Returns:
      A matrix representing the product A * B.
    """
    # Base case: if bitwidth is small, use standard multiplication.
    if w <= threshold:
        return A.dot(B)
    
    # Let l be the lower half bitwidth and k be the remaining bits.
    l = w // 2       # number of bits in the low part
    k = w - l        # number of bits in the high part
    mask = (1 << l) - 1  # mask to extract l low bits

    # Split each element into low and high parts.
    A0 = A & mask         # lower l bits of A
    A1 = A >> l           # remaining high bits of A
    B0 = B & mask         # lower l bits of B
    B1 = B >> l           # high bits of B

    # Recursively compute three products:
    # C1 = product of high parts, computed with bitwidth k
    C1 = kmm_matrix(A1, B1, k, threshold)
    # C0 = product of low parts, computed with bitwidth l
    C0 = kmm_matrix(A0, B0, l, threshold)
    # Cs = product of sums. For safety, we use a bitwidth of max(k, l)+1.
    As = A1 + A0
    Bs = B1 + B0
    C_s = kmm_matrix(As, Bs, max(k, l) + 1, threshold)

    # Combine the three products according to the Karatsuba formula:
    # For two numbers a and b split as a = (A1 << l) + A0 and b = (B1 << l) + B0,
    # we have:
    #   a * b = (C1 << (2*l)) + ((C_s - C1 - C0) << l) + C0.
    #
    # Here, the same idea is applied elementwise (and then via matrix product).
    return (C1 << (2 * l)) + ((C_s - C1 - C0) << l) + C0


def verify_kmm(A, B, w, threshold=8):
    """
    Verifies the KMM multiplication against NumPy's dot product.
    
    Parameters:
      A, B: 2D numpy arrays of integer type.
      w: bitwidth assumed for the elements in A and B.
      threshold: threshold parameter used in kmm_matrix.
    """
    kmm_result = kmm_matrix(A, B, w, threshold)
    expected = A.dot(B)
    
    if np.array_equal(kmm_result, expected):
        print("✔ Karatsuba Matrix Multiplication is correct!")
    else:
        print("✘ Karatsuba Matrix Multiplication is incorrect!")
        print("KMM Result:\n", kmm_result)
        print("Expected Result:\n", expected)


# --- Example Usage ---
# We create two 4x4 matrices whose entries fit in w bits.
# For example, let w = 8 so that each element is an 8-bit number.
A = np.array([[  4,  6,  4,  6],
              [ 12, 14, 12, 14],
              [ 20, 22, 20, 22],
              [ 28, 30, 28, 30]], dtype=np.int64)

B = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]], dtype=np.int64)

# We assume each element is an 8-bit number (values < 256)
w = 8

# Verify the multiplication.
verify_kmm(A + 2**w, B + 2**w, w)
