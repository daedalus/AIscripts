"""
Implementation of the CVM algorithm: Distinct Elements in Streams: An Algorithm for the (Text) Book (2301.10191). 
"""

import random
import math

class F0Estimator:
    def __init__(self, eps, delta, m, n=None):
        # eps: approximation parameter ε
        # delta: failure probability δ
        # m: upper bound on stream length
        # n: domain size (only needed to estimate storage cost)
        self.eps = eps
        self.delta = delta
        self.thresh = math.ceil((2 / (eps**2)) * math.log((8*m)/delta))
        self.X = set()
        self.p = 1.0  # sampling probability p

    def update(self, a):
        # remove old instance of element
        if a in self.X:
            self.X.remove(a)

        # resample it with probability p
        if random.random() < self.p:
            self.X.add(a)

        # if set reaches thresh, downsample
        if len(self.X) == self.thresh:
            # throw each with probability 1/2
            newX = set()
            for x in self.X:
                if random.random() < 0.5:
                    newX.add(x)
            self.X = newX
            self.p /= 2.0

        # If still full, algorithm returns ⊥ in the paper
        if len(self.X) == self.thresh:
            return None  # ⊥

        return True

    def estimate(self):
        # Output |X|/p
        return len(self.X) / self.p

stream = [3,1,2,3,3,2,7,9,1,4,2,5,6,7,8]

est = F0Estimator(eps=0.1, delta=1e-6, m=len(stream))

for a in stream:
    if est.update(a) is None:     # algorithm returned ⊥
        print("Failure (⊥) occurred")
        break

print("Estimated F0:", est.estimate())
print("True F0:", len(set(stream)))

