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

    # Extract sign bits
    sign_a = a_bits >> 31
    sign_b = b_bits >> 31
    sign_res = sign_a ^ sign_b

    # Remove sign bits to get absolute values
    a_abs = a_bits & 0x7FFFFFFF
    b_abs = b_bits & 0x7FFFFFFF

    # Add mantissa+exponent
    added = a_abs + b_abs

    # Subtract offset (0x3F780000 = float representation of 0.96875)
    # This depends on the mantissa bit-width. For 3-bit mantissa, offset = 0x3F800000 - 2**(23-3)
    # But paper suggests fixed value: 0x3F780000
    offset = 0x3F780000
    result = added - offset

    # Ensure result is positive and re-apply sign bit
    result = result & 0x7FFFFFFF
    if sign_res:
        result = result | 0x80000000

    return bits_to_float(result)


a = 3.6
b = 2.1
approx = lmul(a, b)
print("L-Mul approximation:", approx)
print("Actual multiplication:", a * b)
print("Error: ", abs(a*b-approx))