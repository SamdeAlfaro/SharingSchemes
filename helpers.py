from typing import List

# Evaluate a polynomial mod some prime
# given its coefficients and x coordinate
def eval_poly(coeffs: List[int], prime: int, x: int) -> int:
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % prime
    return result
