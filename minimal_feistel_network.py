import hashlib

def feistel_round_function(data_bytes, key_int):
    """Round function that works directly with bytes and integers"""
    key_bytes = key_int.to_bytes(4, byteorder='big')
    combined = data_bytes + key_bytes
    hashed = hashlib.sha256(combined).digest()
    return int.from_bytes(hashed[:4], byteorder='big')  # Use first 32 bits

def feistel_encrypt(block, keys, rounds=4):
    """Encrypt a 64-bit block using a Feistel network"""
    if len(block) != 8:
        raise ValueError("Block must be 8 bytes (64 bits)")
    
    left = block[:4]
    right = block[4:]
    
    for i in range(rounds):
        left_int = int.from_bytes(left, byteorder='big')
        right_int = int.from_bytes(right, byteorder='big')
        
        # Round function uses current right half and key
        round_func_output = feistel_round_function(right, keys[i % len(keys)])
        
        # Feistel operation: new_right = left XOR F(right, key)
        new_right = left_int ^ round_func_output
        
        # Update for next round (swap except in last round)
        left = right
        right = new_right.to_bytes(4, byteorder='big')
    
    # Final output is concatenation of left and right
    return left + right

def feistel_decrypt(block, keys, rounds=4):
    """Decrypt a 64-bit block using a Feistel network"""
    if len(block) != 8:
        raise ValueError("Block must be 8 bytes (64 bits)")
    
    left = block[:4]
    right = block[4:]
    
    for i in reversed(range(rounds)):
        left_int = int.from_bytes(left, byteorder='big')
        right_int = int.from_bytes(right, byteorder='big')
        
        # Round function uses current right half and key
        round_func_output = feistel_round_function(left, keys[i % len(keys)])
        
        # Reverse Feistel operation: new_left = right XOR F(left, key)
        new_left = right_int ^ round_func_output
        
        # Update for next round (swap except in first round)
        right = left
        left = new_left.to_bytes(4, byteorder='big')
    
    # Final output is concatenation of left and right
    return left + right

# Example usage
if __name__ == "__main__":
    # Example keys (32-bit integers)
    keys = [0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0]
    
    # Example 8-byte (64-bit) block to encrypt
    plaintext = b"ABCDEFGH"
    
    print(f"Original: {plaintext}")
    
    # Encrypt
    ciphertext = feistel_encrypt(plaintext, keys)
    print(f"Encrypted: {ciphertext}")
    
    # Decrypt
    decrypted = feistel_decrypt(ciphertext, keys)
    print(f"Decrypted: {decrypted}")
    
    # Verify
    assert decrypted == plaintext, "Decryption failed!"
    print("Success: Decrypted text matches original!")
