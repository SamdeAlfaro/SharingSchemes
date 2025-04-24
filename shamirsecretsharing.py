import datetime
from Crypto.Util.number import getPrime, getRandomRange
from helpers import eval_poly, lagrange_eval
from typing import List, Tuple

class ShamirSecretSharing:
    def __init__(
        self,
        threshold: int,
        num_shares: int,
        bit_length: int = 256,
        # Internal logs for the class
        enable_logs: bool = False
        ):
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")

        self.k = threshold
        self.n = num_shares
        self.bit_length = bit_length
        self.prime = getPrime(bit_length)
        self.enable_logs = enable_logs

    def _random_polynomial(self, secret: int) -> List[int]:
        return [secret] + [getRandomRange(1, self.prime) for _ in range(self.k - 1)]

    def split(self, secret: int) -> List[Tuple[int, int]]:
        start_time = datetime.datetime.now()
        # Split secret into shares.

        if secret >= self.prime:
            raise ValueError("Secret must be less than the prime field")

        poly = self._random_polynomial(secret)
        x_values = set()
        shares = []

        x = 1
        while x <= self.n:
            y = eval_poly(poly, self.prime, x)
            shares.append((x, y))
            x_values.add(x)
            # Simple x values are fine
            x = x + 1
        if self.enable_logs:
            elapsed = datetime.datetime.now() - start_time
            print("split(shares):", elapsed.total_seconds() * 1000)

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        start_time = datetime.datetime.now()
        secret = lagrange_eval(shares, self.prime)
        if self.enable_logs:
            elapsed = datetime.datetime.now() - start_time
            print("reconstruct():", elapsed.total_seconds() * 1000)
        return secret

if __name__ == "__main__":
    secret = 123456798
    print("Secret:", secret)
    sss = ShamirSecretSharing(threshold=3, num_shares=5)

    print("Prime field:", sss.prime)

    shares = sss.split(secret)
    print("Shamir prime bit length:", sss.prime.bit_length())
    print("Generated Shares:")
    for share in shares:
        print(share)

    recovered = sss.reconstruct(shares[:3])
    print("Recovered Secret:", recovered)
    print("Secrets match:", secret == recovered)
