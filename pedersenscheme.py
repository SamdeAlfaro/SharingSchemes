import random
from Crypto.Util.number import getPrime, inverse, GCD
from typing import List, Tuple

def generate_safe_primes(bits=256):
    while True:
        q = getPrime(bits)
        p = 2 * q + 1
        if is_prime(p):
            return p, q

def is_prime(n):
    from sympy import isprime
    return isprime(n)

def find_generator(p, q):
    for g in range(2, p):
        if pow(g, q, p) == 1:
            return g
    raise ValueError("No generator found")

class PedersenSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bits: int = 256):
        self.t = threshold
        self.n = num_shares
        self.p, self.q = generate_safe_primes(bits)
        self.g = find_generator(self.p, self.q)
        self.h = find_generator(self.p, self.q)
        if self.g == self.h:
            self.h = pow(self.g, 2, self.p)

    def split(self, secret: int):
        a = [secret] + [random.randrange(self.q) for _ in range(1, self.t)]
        b = [random.randrange(self.q) for _ in range(self.t)]

        # Generate shares
        shares = []
        for i in range(1, self.n + 1):
            x = i
            s = sum((a[j] * pow(x, j, self.q)) % self.q for j in range(self.t)) % self.q
            r = sum((b[j] * pow(x, j, self.q)) % self.q for j in range(self.t)) % self.q
            shares.append((x, s, r))

        commitments = [pow(self.g, a_j, self.p) * pow(self.h, b_j, self.p) % self.p for a_j, b_j in zip(a, b)]

        return shares, commitments

    def verify_share(self, x: int, s: int, r: int, commitments: List[int]) -> bool:
        lhs = (pow(self.g, s, self.p) * pow(self.h, r, self.p)) % self.p
        rhs = 1
        for j, C_j in enumerate(commitments):
            rhs = (rhs * pow(C_j, pow(x, j, self.q), self.p)) % self.p
        return lhs == rhs

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """Lagrange interpolation in Z_q to recover secret."""
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
