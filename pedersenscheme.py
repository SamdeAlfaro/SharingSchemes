from typing import List, Tuple
from Crypto.Util.number import getPrime, inverse
from Crypto.Random import random
from sympy import isprime
from shamirsecretsharing import ShamirSecretSharing

def generate_group(bits=256):
    from sympy import isprime  # Ensure you're using sympy's isprime

    while True:
        q = getPrime(bits)
        for _ in range(1000):  # Try up to 1000 k values for this q
            k = random.randint(2, 1 << 12)  # Larger range for k
            p = k * q + 1
            if isprime(p):
                # Now find g, h in subgroup of order q
                def find_subgroup_generator(p, q, k):
                    while True:
                        g = random.randrange(2, p - 1)
                        # Ensure g not in subgroup of small order
                        if pow(g, (p - 1) // q, p) != 1:
                            return pow(g, k, p)  # g^k mod p has order q

                g = find_subgroup_generator(p, q, k)
                h = find_subgroup_generator(p, q, k)
                while h == g:
                    h = find_subgroup_generator(p, q, k)
                return p, q, g, h

class PedersenSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bits: int = 256):
        self.t = threshold
        self.n = num_shares
        self.bits = bits

        # Generate primes and generators
        self.p, self.q, self.g, self.h = generate_group(bits)


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
        # Maybe we break this out and share split with Shamir and Feldman.
        # A lot of these functions can be shared between them if we want.
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
        print("PRIME:" + str(self.q))
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
    print("Secret:", secret)
    pss = PedersenSecretSharing(threshold=3, num_shares=5)

    print("p:", pss.p)
    print("q:", pss.q)
    print("g:", pss.g)
    print("h:", pss.h)

    shares, commitments = pss.split(secret)
    print("Pedersen q bit length:", pss.q.bit_length())

    print("\nShares (x, s, r):")
    for x, s, r in shares:
        print(f"{x}: s = {s}, r = {r}, valid? {pss.verify_share(x, s, r, commitments)}")

    recovered = pss.reconstruct([(x, s) for x, s, _ in shares[:3]])
    print("\nRecovered Secret:", recovered)
