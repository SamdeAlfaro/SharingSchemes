from typing import List, Tuple
from Crypto.Util.number import getPrime
from helpers import eval_poly, lagrange_eval
from Crypto.Random import random
from sympy import isprime

def generate_group(bits=256):
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
                return p, q, g

class FeldmanVerifiableSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bits: int = 256):
        self.t = threshold
        self.n = num_shares
        self.bits = bits

        self.p, self.q, self.g = generate_group(bits)

    def split(self, secret: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        # Coefficients for the secret-sharing polynomial
        coeffs = [secret] + [random.randrange(self.q) for _ in range(1, self.t)]

        # Shares (no r)
        shares = []
        for x in range(1, self.n + 1):
            y = eval_poly(coeffs, self.q, x)
            shares.append((x, y))

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
        return lagrange_eval(shares, self.q)

if __name__ == "__main__":
    secret = 123456789
    print("Secret:", secret)
    feldman = FeldmanVerifiableSecretSharing(threshold=3, num_shares=5)

    print("p:", feldman.p)
    print("q:", feldman.q)
    print("g:", feldman.g)

    shares, commitments = feldman.split(secret)
    print("Feldman q bit length:", feldman.q.bit_length())

    print("Shares (x, y):")
    for x, y in shares:
        is_valid = feldman.verify_share(x, y, commitments)
        print(f"{x}: y = {y}, is_valid = {is_valid}")

    recovered = feldman.reconstruct(shares[:3])
    print("Recovered Secret:", recovered)
    print("Secrets match:", secret == recovered)
