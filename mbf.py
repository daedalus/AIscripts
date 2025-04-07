import mmh3
import bitarray
import math

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray.bitarray(size)
        self.bit_array.setall(0)
        if size <= 1<<128:
            self.hashfunc = mmh3.hash128
        if size <= 1<<64:
            self.hashfunc = mmh3.hash64
        if size <= 1<<32:
            self.hashfunc = mmh3.hash

    def add(self, item):
        for seed in range(self.hash_count):
            bit_pos = self.hashfunc(item, seed, False) % self.size
            self.bit_array[bit_pos] = 1

    def lookup(self, item):
        for seed in range(self.hash_count):
            bit_pos = self.hashfunc(item, seed, False) % self.size
            if not self.bit_array[bit_pos]:
                return False
        return True

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
print(bf.hashfunc)
