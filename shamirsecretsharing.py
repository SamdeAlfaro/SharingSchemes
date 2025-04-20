from Crypto.Util.number import getPrime, getRandomRange, inverse
from typing import List, Tuple
import hashlib
from sympy import isprime

def generate_safe_primes(bits=256):
    while True:
        q = getPrime(bits)
        p = 2 * q + 1
        if isprime(p):
            return p, q

# For Shamir, you can adopt a similar pattern to generate a safe prime for consistency


class ShamirSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bit_length: int = 256):
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        
        self.k = threshold
        self.n = num_shares
        self.bit_length = bit_length
        self.prime, _ = generate_safe_primes(bits=256)

    def _hash_secret(self, secret_bytes: bytes) -> int:
        # Hash secret into the field.
        digest = hashlib.sha256(secret_bytes).digest()
        return int.from_bytes(digest, 'big') % self.prime

    def _random_polynomial(self, secret: int) -> List[int]:
        return [secret] + [getRandomRange(1, self.prime) for _ in range(self.k - 1)]

    def _evaluate_polynomial(self, coeffs: List[int], x: int) -> int:
        #  Evaluate polynomial at point x (mod prime)
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % self.prime
        return result

    def split(self, secret: int, hash_secret: bool = False) -> List[Tuple[int, int]]:
        # Split secret into shares.
        if hash_secret:
            secret = self._hash_secret(str(secret).encode())

        if secret >= self.prime:
            raise ValueError("Secret must be less than the prime field")
        
        poly = self._random_polynomial(secret)
        x_values = set()
        shares = []

        while len(x_values) < self.n:
            x = getRandomRange(1, self.prime)
            if x not in x_values:
                y = self._evaluate_polynomial(poly, x)
                shares.append((x, y))
                x_values.add(x)

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            li = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    li *= (xj * inverse(xj - xi, self.prime)) % self.prime
                    li %= self.prime
            secret += yi * li
            secret %= self.prime
        return secret

if __name__ == "__main__":
    secret = 98765432109876543210
    sss = ShamirSecretSharing(threshold=3, num_shares=5, bit_length=256)

    print("Prime field:", sss.prime)

    shares = sss.split(secret)
    print("Generated Shares:")
    for share in shares:
        print(share)

    recovered = sss.reconstruct(shares[:3])
    print("Recovered Secret:", recovered)
