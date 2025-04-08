from Crypto.Util.number import getPrime, getRandomRange, inverse
from typing import List, Tuple

class BlakleySecretSharing:
    def __init__(self, threshold: int, num_shares: int, bit_length: int = 256):
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        self.k = threshold
        self.n = num_shares
        self.p = getPrime(bit_length)  # finite field modulus

    def split(self, secret: int) -> List[Tuple[List[int], int]]:
        """Split the secret using hyperplanes (one per share)."""
        if secret >= self.p:
            raise ValueError("Secret must be less than the prime field")

        # The secret is embedded as the first coordinate in a k-dimensional vector
        secret_vector = [secret] + [getRandomRange(0, self.p) for _ in range(self.k - 1)]

        shares = []
        for _ in range(self.n):
            # Generate a random hyperplane: vector a and scalar b such that a·x = b
            coeffs = [getRandomRange(0, self.p) for _ in range(self.k)]
            b = sum(c * s for c, s in zip(coeffs, secret_vector)) % self.p
            shares.append((coeffs, b))

        return shares

    def reconstruct(self, shares: List[Tuple[List[int], int]]) -> int:
        """Reconstruct the secret from k shares (solve system of linear equations)."""
        if len(shares) < self.k:
            raise ValueError(f"Need at least {self.k} shares to reconstruct the secret")

        # Extract matrix A and vector b
        A = [row[0] for row in shares[:self.k]]
        b = [row[1] for row in shares[:self.k]]

        # Solve A * x = b mod p
        x = self._solve_linear_system_mod_p(A, b, self.p)

        return x[0]  # The secret is the first coordinate

    def _solve_linear_system_mod_p(self, A: List[List[int]], b: List[int], p: int) -> List[int]:
        """Solves Ax = b mod p using Gaussian elimination"""
        n = len(A)
        A = [row[:] for row in A]
        b = b[:]

        for i in range(n):
            # Find pivot
            if A[i][i] == 0:
                for j in range(i + 1, n):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]
                        b[i], b[j] = b[j], b[i]
                        break
                else:
                    raise ValueError("Matrix is singular")

            # Normalize pivot row
            inv = inverse(A[i][i], p)
            A[i] = [(val * inv) % p for val in A[i]]
            b[i] = (b[i] * inv) % p

            # Eliminate below
            for j in range(i + 1, n):
                factor = A[j][i]
                A[j] = [(a - factor * ai) % p for a, ai in zip(A[j], A[i])]
                b[j] = (b[j] - factor * b[i]) % p

        # Back substitution
        x = [0] * n
        for i in reversed(range(n)):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] = (x[i] - A[i][j] * x[j]) % p

        return x

if __name__ == "__main__":
    secret = 123456789
    bss = BlakleySecretSharing(threshold=3, num_shares=5)

    print("Prime field:", bss.p)

    shares = bss.split(secret)
    print("\nShares (coeffs, b):")
    for coeffs, b_val in shares:
        print(f"{coeffs} = {b_val}")

    recovered = bss.reconstruct(shares[:3])
    print("\nRecovered Secret:", recovered)