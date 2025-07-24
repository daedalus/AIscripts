import sys
import os
import math
import hashlib
import mmap
from typing import Optional
from pybloom_live import BloomFilter

# AES key sizes and schedule lengths in bytes
AES128_KEY_SIZE = 16
AES192_KEY_SIZE = 24
AES256_KEY_SIZE = 32

AES128_KEY_SCHEDULE_SIZE = 176
AES192_KEY_SCHEDULE_SIZE = 208
AES256_KEY_SCHEDULE_SIZE = 240

BUFFER_SIZE = 10 * 1024 * 1024  # 10 MB
WINDOW_SIZE = AES256_KEY_SCHEDULE_SIZE

def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    length = len(data)
    return -sum((f / length) * math.log2(f / length) for f in freq if f > 0)

# Entropy filter with two modes
def entropy(buffer: bytes, first: bool = False, mode: str = "legacy") -> bool:
    if mode == "shannon":
        return shannon_entropy(buffer) > 7.0

    static = getattr(entropy, "_state", {"count": [0] * 256, "first_entropy": True})
    if first:
        static["first_entropy"] = True

    if static["first_entropy"]:
        static["first_entropy"] = False
        static["count"] = [0] * 256
        for b in buffer:
            static["count"][b] += 1

    for count in static["count"]:
        if count > 8:
            return True

    if len(buffer) > 1:
        static["count"][buffer[0]] -= 1
        static["count"][buffer[-1]] += 1

    entropy._state = static
    return False

def display_key(key: bytes):
    print(" ".join(f"{byte:02x}" for byte in key))

# Stub key schedule matchers
def valid_aes128_schedule(data: bytes) -> bool:
    return False

def valid_aes192_schedule(data: bytes) -> bool:
    return False

def valid_aes256_schedule(data: bytes) -> bool:
    return False

def scan_buffer(buffer: bytes, size: int, offset: int, entropy_mode: str = "legacy"):
    for pos in range(size):
        current = buffer[pos:pos + AES256_KEY_SCHEDULE_SIZE]
        if len(current) < AES256_KEY_SCHEDULE_SIZE:
            break
        first = (offset + pos == 0)

        if entropy(current[:AES128_KEY_SCHEDULE_SIZE], first, entropy_mode):
            continue

        if valid_aes128_schedule(current):
            print(f"Found AES-128 key schedule at offset 0x{offset + pos:x}:")
            display_key(current[:AES128_KEY_SIZE])
        if valid_aes192_schedule(current):
            print(f"Found AES-192 key schedule at offset 0x{offset + pos:x}:")
            display_key(current[:AES192_KEY_SIZE])
        if valid_aes256_schedule(current):
            print(f"Found AES-256 key schedule at offset 0x{offset + pos:x}:")
            display_key(current[:AES256_KEY_SIZE])

def scan_file(filename: str, entropy_mode: str = "legacy", use_mmap: bool = False,
              bloom: Optional[BloomFilter] = None, hash_table: Optional[set] = None) -> bool:
    try:
        with open(filename, "rb") as f:
            print(f"Searching {filename}")

            if use_mmap:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                size = len(mm)
                for offset in range(0, size, BUFFER_SIZE):
                    chunk = mm[offset:offset + BUFFER_SIZE]
                    if not chunk:
                        break

                    chunk_hash = hashlib.sha256(chunk).hexdigest()
                    if bloom:
                        if chunk_hash in bloom:
                            print(f"Duplicate chunk hash at offset 0x{offset:x}")
                        else:
                            bloom.add(chunk_hash)
                    if hash_table is not None:
                        hash_table.add(chunk_hash)

                    scan_buffer(chunk, len(chunk), offset, entropy_mode)
                mm.close()

            else:
                offset = 0
                buffer = bytearray(WINDOW_SIZE + BUFFER_SIZE)

                while True:
                    f.seek(offset)
                    read_bytes = f.read(BUFFER_SIZE)
                    if not read_bytes:
                        break

                    chunk_hash = hashlib.sha256(read_bytes).hexdigest()
                    if bloom:
                        if chunk_hash in bloom:
                            print(f"Duplicate chunk hash at offset 0x{offset:x}")
                        else:
                            bloom.add(chunk_hash)
                    if hash_table is not None:
                        hash_table.add(chunk_hash)

                    buffer[WINDOW_SIZE:] = read_bytes
                    total_size = len(read_bytes)
                    size = total_size if offset == 0 else total_size + WINDOW_SIZE

                    if offset == 0:
                        scan_buffer(buffer[WINDOW_SIZE:], total_size, 0, entropy_mode)
                    else:
                        scan_buffer(buffer[:WINDOW_SIZE + total_size], total_size, offset - WINDOW_SIZE, entropy_mode)

                    offset += total_size
                    buffer[:WINDOW_SIZE] = buffer[BUFFER_SIZE:BUFFER_SIZE + WINDOW_SIZE]

        return False
    except Exception as e:
        print(f"Error scanning {filename}: {e}")
        return True

def main():
    if len(sys.argv) < 2:
        print("FindAES (Python version)")
        print("Searches for AES-128, AES-192, and AES-256 key schedules in files")
        print("Usage: findaes.py [--shannon] [--mmap] FILES")
        return

    entropy_mode = "legacy"
    use_mmap = False
    files = []

    bloom = BloomFilter(capacity=1000000, error_rate=0.001)
    hash_table = set()

    for arg in sys.argv[1:]:
        if arg == "--shannon":
            entropy_mode = "shannon"
        elif arg == "--mmap":
            use_mmap = True
        else:
            files.append(arg)

    for filename in files:
        scan_file(filename, entropy_mode, use_mmap, bloom, hash_table)

    # Optional: Save hashes to file
    # with open("hashes.txt", "w") as out:
    #     for h in sorted(hash_table):
    #         out.write(f"{h}\n")

if __name__ == "__main__":
    main()