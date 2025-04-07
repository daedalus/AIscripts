import mmh3
import random

class CuckooFilter:
    def __init__(self, capacity, bucket_size=4, fingerprint_size=8, max_kicks=500):
        """
        Initialize the Cuckoo Filter.
        
        Parameters:
        - capacity: Expected number of items to store
        - bucket_size: Number of fingerprints per bucket (default: 4)
        - fingerprint_size: Size of fingerprint in bits (default: 8)
        - max_kicks: Maximum number of relocations before giving up (default: 500)
        """
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks
        self.size = self._next_power_of_two(capacity // bucket_size)
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        
    def _next_power_of_two(self, n):
        """Return the next power of two greater than or equal to n"""
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    def _get_fingerprint(self, item):
        """Generate fingerprint for an item"""
        fp_hash = mmh3.hash_bytes(item)[0]  # Use first byte for fingerprint
        mask = (1 << self.fingerprint_size) - 1
        return (fp_hash & mask) or 1  # Ensure fingerprint is never 0
    
    def _get_index(self, item):
        """Get primary index for an item"""
        return mmh3.hash(item) % self.size
    
    def _get_alt_index(self, fingerprint, index):
        """Get alternate index using fingerprint"""
        return (index ^ mmh3.hash(str(fingerprint).encode())) % self.size
    
    def add(self, item):
        """Add an item to the filter. Returns True if successful, False otherwise"""
        if self.count >= self.capacity:
            return False
            
        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)
        
        # Try to insert in primary or secondary bucket
        if self._insert_fingerprint(i1, fingerprint):
            self.count += 1
            return True
        if self._insert_fingerprint(i2, fingerprint):
            self.count += 1
            return True
            
        # Both buckets full, need to relocate existing fingerprints
        return self._relocate_and_add(i1, i2, fingerprint)
    
    def _insert_fingerprint(self, index, fingerprint):
        """Insert fingerprint into bucket if space available"""
        if len(self.buckets[index]) < self.bucket_size:
            self.buckets[index].append(fingerprint)
            return True
        return False
    
    def _relocate_and_add(self, i1, i2, fingerprint):
        """Relocate fingerprints to make space for new item"""
        # Randomly select one of the two buckets to kick from
        current_index = random.choice([i1, i2])
        
        for _ in range(self.max_kicks):
            # Select random fingerprint from current bucket to relocate
            if not self.buckets[current_index]:
                continue
                
            victim_pos = random.randint(0, len(self.buckets[current_index]) - 1)
            victim_fp = self.buckets[current_index][victim_pos]
            
            # Remove victim fingerprint
            del self.buckets[current_index][victim_pos]
            
            # Try to insert the new fingerprint in its place
            if self._insert_fingerprint(current_index, fingerprint):
                self.count += 1
                return True
                
            # If that failed, find alternate location for victim fingerprint
            new_index = self._get_alt_index(victim_fp, current_index)
            
            # Try to insert victim fingerprint in its alternate location
            if self._insert_fingerprint(new_index, victim_fp):
                self.count += 1
                return True
                
            # If that failed, continue with the victim fingerprint
            fingerprint = victim_fp
            current_index = new_index
            
        # If we've exhausted max kicks, return failure
        return False
    
    def contains(self, item):
        """Check if item is probably in the filter (may have false positives)"""
        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)
        
        return fingerprint in self.buckets[i1] or fingerprint in self.buckets[i2]
    
    def remove(self, item):
        """Remove an item from the filter. Returns True if successful, False otherwise"""
        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)
        
        if fingerprint in self.buckets[i1]:
            self.buckets[i1].remove(fingerprint)
            self.count -= 1
            return True
        elif fingerprint in self.buckets[i2]:
            self.buckets[i2].remove(fingerprint)
            self.count -= 1
            return True
        return False
    
    def __contains__(self, item):
        """Support for 'in' operator"""
        return self.contains(item)
    
    def __len__(self):
        """Return number of items in filter"""
        return self.count
