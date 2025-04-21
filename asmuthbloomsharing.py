from sympy import nextprime
import secrets  # Use the cryptographically secure secrets module
from math import prod

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
        if type(first_prime) != int:
            raise ValueError("first_prime: {first_prime} not an int")
        multiplier = 10
        while True:
            sequence = [first_prime]
            p = nextprime(s * multiplier)
            for _ in range(self.n_holder):
                p = nextprime(p)
                if type(p) != int:
                    raise ValueError(f"p: {p} not an int")
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

    # Different random number generators can
    # affect perf so maybe we want to share them
    def _generate_random(self):
        max_r = (self.M - self.secret) // self.sequence[0]
        self.random_r = secrets.randbelow(max_r) + 1  # Use secrets.randbelow() for cryptographic security
        self.y = self.secret + self.random_r * self.sequence[0]
        if self.verbose:
            print(f"Random r: {self.random_r}")
            print(f"y: {self.y}")

    def get_holders(self):
        return self.holders

    def get_sequence(self):
        return self.sequence

    def solve(self):
        # Securely generate random indices for shuffling
        random_indices = [secrets.randbelow(2**32) for _ in range(len(self.holders))]

        # Pair each holder with a random index, then sort by the random index
        indexed_holders = list(zip(self.holders, random_indices))
        indexed_holders.sort(key=lambda x: x[1])  # Sort by the random index

        # Select the first min_holder holders
        chosen = [h[0] for h in indexed_holders[:self.min_holder]]
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

if __name__ == "__main__":
    # Example test
    ab = AsmuthBloom(n_holders=5, min_holder=3, secret=123456789, verbose=True)
    recovered = ab.solve()
