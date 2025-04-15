# https://arxiv.org/abs/2410.00907


import struct

def float_to_bits(f):
    """Convert a 32-bit float to its bit representation (as an int)."""
    return struct.unpack('>I', struct.pack('>f', f))[0]

def bits_to_float(b):
    """Convert a 32-bit int bit representation back to float."""
    return struct.unpack('>f', struct.pack('>I', b))[0]

def lmul(a, b):
    """
    Approximate multiplication of two 32-bit floats using L-Mul algorithm.
    This implementation replicates the unsigned integer addition behavior.
    """
    # Convert to 32-bit integer representations
    a_bits = float_to_bits(a)
    b_bits = float_to_bits(b)

    # Combined sign calculation and absolute value masks
    sign_mask = 0x80000000
    abs_mask = 0x7FFFFFFF
    offset = 0x3F780000  # Precomputed constant

    # Efficient sign handling using XOR and shift
    sign = ((a_bits ^ b_bits) & sign_mask)
    
    # Core L-Mul operation
    result = (
        ((a_bits & abs_mask) + 
         (b_bits & abs_mask) - 
         offset
        ) & abs_mask | sign)

    return bits_to_float(result)


a = 3.6
b = 2.1
approx = lmul(a, b)
print("L-Mul approximation:", approx)
print("Actual multiplication:", a * b)
print("Error: ", abs(a*b-approx))
