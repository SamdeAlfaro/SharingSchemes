from typing import List, Tuple
from Crypto.Util.number import getPrime, inverse
from Crypto.Random import random
from sympy import isprime

def generate_safe_primes(bits=256):
    while True:
        q = getPrime(bits)
        p = 2 * q + 1
        if isprime(p):
            return p, q

def find_generator(p, q):
    for g in range(2, p):
        if pow(g, q, p) == 1 and pow(g, 2, p) != 1:
            return g
    raise ValueError("No generator found")

class FeldmanVerifiableSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bits: int = 256):
        self.t = threshold
        self.n = num_shares
        self.bits = bits

        # Safe prime setup: p = 2q + 1
        self.p, self.q = generate_safe_primes(bits)
        self.g = find_generator(self.p, self.q)

    def _eval_poly(self, coeffs: List[int], x: int) -> int:
        return sum((coeff * pow(x, i, self.q)) % self.q for i, coeff in enumerate(coeffs)) % self.q

    def split(self, secret: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        # Coefficients for the secret-sharing polynomial
        coeffs = [secret] + [random.randrange(self.q) for _ in range(1, self.t)]

        # Shares (no r)
        shares = []
        for x in range(1, self.n + 1):
            y = self._eval_poly(coeffs, x)
            shares.append((x, y))

        # Commitments: C_j = g^a_j mod p
        commitments = [pow(self.g, a, self.p) for a in coeffs]

        return shares, commitments

    def verify_share(self, x: int, y: int, commitments: List[int]) -> bool:
        # Check that g^y == product of C_j^{x^j}
        lhs = pow(self.g, y, self.p)
        rhs = 1
        for j, C_j in enumerate(commitments):
            rhs = (rhs * pow(C_j, pow(x, j, self.q), self.p)) % self.p
        return lhs == rhs

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        # Standard Lagrange interpolation over Z_q
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            li = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    li *= (xj * inverse(xj - xi, self.q)) % self.q
                    li %= self.q
            secret += yi * li
            secret %= self.q
        return secret

if __name__ == "__main__":
    secret = 123456789
    feldman = FeldmanVerifiableSecretSharing(threshold=3, num_shares=5)

    print("p:", feldman.p)
    print("q:", feldman.q)
    print("g:", feldman.g)

    shares, commitments = feldman.split(secret)

    print("\nShares (x, y):")
    for x, y in shares:
        print(f"{x}: y = {y}, valid? {feldman.verify_share(x, y, commitments)}")

    recovered = feldman.reconstruct(shares[:3])
    print("\nRecovered Secret:", recovered)
