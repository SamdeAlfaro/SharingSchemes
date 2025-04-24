from typing import List, Tuple
from Crypto.Util.number import inverse

# Evaluate a polynomial mod some prime
# given its coefficients and x coordinate
def eval_poly(coeffs: List[int], prime: int, x: int) -> int:
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % prime
    return result

# Build the unique polynomial evaluating to zero
def lagrange_eval(ys: List[Tuple[int, int]], prime: int):
    secret = 0
    for i, (xi, yi) in enumerate(ys):
        li = 1
        for j, (xj, _) in enumerate(ys):
            if i != j:
                li *= (xj * inverse(xj - xi, prime)) % prime
                li %= prime
        secret += yi * li
        secret %= prime
    return secret
