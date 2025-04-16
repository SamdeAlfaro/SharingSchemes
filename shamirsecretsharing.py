from Crypto.Util.number import getPrime, getRandomRange, inverse
from typing import List, Tuple
import hashlib

class ShamirSecretSharing:
    def __init__(self, threshold: int, num_shares: int, bit_length: int = 256):
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        
        self.k = threshold
        self.n = num_shares
        self.bit_length = bit_length
        self.prime = getPrime(bit_length)

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

    def _lagrange_interpolate(self, x: int, x_s: List[int], y_s: List[int]) -> int:
        # Perform Lagrange interpolation at point x.
        total = 0
        k = len(x_s)
        for i in range(k):
            xi, yi = x_s[i], y_s[i]
            prod = 1
            for j in range(k):
                if i != j:
                    xj = x_s[j]
                    denom = (xi - xj) % self.prime
                    numer = (x - xj) % self.prime
                    prod = (prod * numer * inverse(denom, self.prime)) % self.prime
            total = (total + yi * prod) % self.prime
        return total

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        # Reconstruct the secret from k shares.
        if len(shares) < self.k:
            raise ValueError(f"Need at least {self.k} shares to reconstruct the secret")

        x_s, y_s = zip(*shares[:self.k])
        if len(set(x_s)) < self.k:
            raise ValueError("Duplicate x-values in shares")
        
        return self._lagrange_interpolate(0, list(x_s), list(y_s))


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
