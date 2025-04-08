from sympy import nextprime
import random
from functools import reduce
from math import prod

# Derived from; https://github.com/AlamHasabie/simple-asmuth-bloom/blob/master/asmuth_bloom.py

def chinese_remainder(moduli, residues):
    """Solve the system of congruences using the Chinese Remainder Theorem."""
    total = 0
    M = prod(moduli)
    for m_i, r_i in zip(moduli, residues):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)
        total += r_i * M_i * inv
    return total % M

class Holder:
    def __init__(self):
        self.modulo = 0
        self.secret = 0

    def __str__(self):
        return f"Modulo = {self.modulo} | Secret = {self.secret}"

# Asmuth-Bloom implementation
class AsmuthBloom:
    def __init__(self, n_holders, min_holder, secret, verbose=False):
        self.verbose = verbose
        self.n_holder = n_holders
        self.min_holder = min_holder
        self.secret = secret
        self.holders = [Holder() for _ in range(n_holders)]
        self._generate_holders()

    def _generate_holders(self):
        self._generate_sequence()
        self._generate_random()
        for i in range(self.n_holder):
            self.holders[i].modulo = self.sequence[i + 1]
            self.holders[i].secret = self.y % self.holders[i].modulo

    def _generate_sequence(self):
        s = self.secret
        first_prime = nextprime(s)
        multiplier = 10
        while True:
            sequence = [first_prime]
            p = nextprime(s * multiplier)
            for _ in range(self.n_holder):
                p = nextprime(p)
                sequence.append(p)
            if self._is_sequence_valid(sequence):
                self.sequence = sequence
                self.M = prod(sequence[1:self.min_holder + 1])
                if self.verbose:
                    print(f"Valid sequence: {self.sequence}")
                    print(f"Big M: {self.M}")
                return
            multiplier += 1

    def _is_sequence_valid(self, seq):
        lower = prod(seq[1:self.min_holder + 1])
        upper = seq[0] * prod(seq[-(self.min_holder - 1):])
        return lower > upper

    def _generate_random(self):
        max_r = (self.M - self.secret) // self.sequence[0]
        self.random_r = random.randint(1, max_r)
        self.y = self.secret + self.random_r * self.sequence[0]
        if self.verbose:
            print(f"Random r: {self.random_r}")
            print(f"y: {self.y}")

    def get_holders(self):
        return self.holders

    def get_sequence(self):
        return self.sequence

    def solve(self):
        chosen = random.sample(self.holders, self.min_holder)
        moduli = [h.modulo for h in chosen]
        remainders = [h.secret for h in chosen]
        solution = chinese_remainder(moduli, remainders)
        recovered = solution % self.sequence[0]
        if self.verbose:
            print("Chosen holders:")
            for h in chosen:
                print(h)
            print(f"Recovered y: {solution}, Secret: {recovered}")
        return recovered

# Example test
ab = AsmuthBloom(n_holders=5, min_holder=3, secret=123456789, verbose=True)
recovered = ab.solve()
recovered
