from sympy import nextprime
import secrets
from math import prod

def chinese_remainder(moduli, residues):
    total = 0
    M = prod(moduli)
    for m_i, r_i in zip(moduli, residues):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)
        total += r_i * M_i * inv
    return total % M

class AsmuthBloom:
    def __init__(self, threshold, n_shares):
        self.n = n_shares
        self.k = threshold
        self.sequence = []
        self.M = None
        self.y = None
        self.random_r = None

    def _is_valid_sequence(self, seq):
        lower = prod(seq[1:self.k+1])
        upper = seq[0] * prod(seq[-(self.k - 1):])
        return lower > upper

    def _generate_moduli_sequence(self, secret):
        m0 = nextprime(secret)
        multiplier = 10
        while True:
            sequence = [m0]
            p = nextprime(secret * multiplier)
            for _ in range(self.n):
                p = nextprime(p)
                sequence.append(p)

            if self._is_valid_sequence(sequence):
                self.sequence = sequence
                self.M = prod(sequence[1:self.k+1])
                return
            multiplier += 1

    def _generate_y(self, secret):
        m0 = self.sequence[0]
        max_r = (self.M - secret) // m0
        self.random_r = secrets.randbelow(max_r) + 1
        self.y = secret + self.random_r * m0

    def split(self, secret):
        self.secret = secret
        self._generate_moduli_sequence(secret)
        self._generate_y(secret)
        shares = []
        for i in range(self.n):
            m_i = self.sequence[i+1]
            s_i = self.y % m_i
            shares.append((m_i, s_i))
        return shares

    def reconstruct(self, shares):
        moduli, residues = zip(*shares[:self.k])
        y_recovered = chinese_remainder(moduli, residues)
        secret_recovered = y_recovered % self.sequence[0]
        return secret_recovered

if __name__ == "__main__":
    ab = AsmuthBloom(threshold=3, n_shares=5)
    secret = 123456789
    shares = ab.split(secret)
    print("\nGenerated shares:")
    for i, (m, s) in enumerate(shares):
        print(f"Share {i+1}: (mod {m}) -> {s}")

    recovered = ab.reconstruct(shares)
    print(f"\nRecovered secret: {recovered}")