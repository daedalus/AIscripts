import mmh3
import bitarray
import math

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray.bitarray(size)
        self.bit_array.setall(0)

    def _get_hash_indices(self, item):
        # Get 128-bit hash (as a single integer)
        hash128 = mmh3.hash128(item, signed=False)
        
        # Split into 4x 32-bit parts (or 2x 64-bit)
        hash_parts = [
            (hash128 >> 96) & 0xFFFFFFFF,  # First 32 bits
            (hash128 >> 64) & 0xFFFFFFFF,  # Next 32 bits
            (hash128 >> 32) & 0xFFFFFFFF,  # Next 32 bits
            hash128 & 0xFFFFFFFF           # Last 32 bits
        ]
        
        # Generate `hash_count` indices (use modulo to stay within bounds)
        return [hash_parts[i % 4] % self.size for i in range(self.hash_count)]

    def add(self, item):
        for index in self._get_hash_indices(item):
            self.bit_array[index] = 1

    def lookup(self, item):
        return all(self.bit_array[index] for index in self._get_hash_indices(item))

    @classmethod
    def from_expected_items(cls, n, p):
        size = - (n * math.log(p)) / (math.log(2) ** 2)
        hash_count = (size / n) * math.log(2)
        return cls(int(size), int(hash_count))

    def false_positive_probability(self, num_items):
        return (1 - (1 - 1 / self.size) ** (self.hash_count * num_items)) ** self.hash_count

# Example Usage
bf = BloomFilter.from_expected_items(100000, 0.01)  # 100k items, 1% false positives
bf.add("example_item")
print(bf.lookup("example_item"))  # True
print(bf.lookup("nonexistent"))   # False (1% chance of True)
