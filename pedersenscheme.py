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

class PedersenSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bits: int = 256):
        self.t = threshold
        self.n = num_shares
        self.bits = bits

        # Generate safe primes and generators
        self.p, self.q = generate_safe_primes(bits)
        self.g = find_generator(self.p, self.q)
        self.h = find_generator(self.p, self.q)
        if self.g == self.h:
            self.h = pow(self.g, 2, self.p)

    def eval_poly(self, poly: List[int], x: int) -> int:
        return sum((coeff * pow(x, i, self.q)) % self.q for i, coeff in enumerate(poly)) % self.q

    def split(self, secret: int) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        # Generate the secret and blinding polynomials over Z_q
        a_coeffs = [secret] + [random.randrange(self.q) for _ in range(1, self.t)]
        b_coeffs = [random.randrange(self.q) for _ in range(self.t)]

        # Evaluate the shares
        shares = []
        for x in range(1, self.n + 1):
            s = self.eval_poly(a_coeffs, x)
            r = self.eval_poly(b_coeffs, x)
            shares.append((x, s, r))

        # Commitments: C_j = g^a_j * h^b_j mod p
        commitments = [
            (pow(self.g, a, self.p) * pow(self.h, b, self.p)) % self.p
            for a, b in zip(a_coeffs, b_coeffs)
        ]

        return shares, commitments

    def verify_share(self, x: int, s: int, r: int, commitments: List[int]) -> bool:
        lhs = (pow(self.g, s, self.p) * pow(self.h, r, self.p)) % self.p
        rhs = 1
        for j, C_j in enumerate(commitments):
            rhs = (rhs * pow(C_j, pow(x, j, self.q), self.p)) % self.p
        return lhs == rhs

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        # Lagrange interpolation in Z_q to recover secret.
        secret = 0
        for i, (xi, si) in enumerate(shares):
            li = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    li *= (xj * inverse(xj - xi, self.q)) % self.q
                    li %= self.q
            secret += si * li
            secret %= self.q
        return secret

if __name__ == "__main__":
    secret = 123456789
    pss = PedersenSecretSharing(threshold=3, num_shares=5)

    print("p:", pss.p)
    print("q:", pss.q)
    print("g:", pss.g)
    print("h:", pss.h)

    shares, commitments = pss.split(secret)

    print("\nShares (x, s, r):")
    for x, s, r in shares:
        print(f"{x}: s = {s}, r = {r}, valid? {pss.verify_share(x, s, r, commitments)}")

    recovered = pss.reconstruct([(x, s) for x, s, r in shares[:3]])
    print("\nRecovered Secret:", recovered)
