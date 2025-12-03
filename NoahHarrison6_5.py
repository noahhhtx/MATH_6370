# This script utilizes Newton's Method to compute the root of a function.

import sys
import fixedpoint_function as f

def newton_method(x0, tol = (10**-5), h = 0.0001, nmax = 1000):

    x = x0
    r = sys.maxsize * 2 + 1
    i = 1

    while r > tol and i <= nmax:

        x_new = f.newton(x, h)
        r = abs(x_new - x)
        print(f"Iteration {i}: {r}")
        x = x_new
        i += 1

    return x

if __name__ == "__main__":

    x = newton_method(3)
    print("Root is", x)
