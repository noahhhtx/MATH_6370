# This script utilizes the Bisection Algorithm to compute the root of a function.

import sys
import rootfinding_function as f

def bisection(a, b, tol = (10**-5), maxiter = 1000):

    r = sys.maxsize * 2 + 1

    a1 = a
    b1 = b
    x = (a1 + b1)/2
    i = 1

    while r > tol and i <= maxiter:

        if f.func(a1) * f.func(x) < 0:
            b1 = x
        elif f.func(x) * f.func(b1) < 0:
            a1 = x
        else:
            print("Root found.")
            return x

        x_new = (a1 + b1) / 2
        r = abs(x_new - x)
        print(f"Iteration {i}: {r}")
        x = x_new
        i += 1

    return x

if __name__ == "__main__":

    x = bisection(1, 3)
    print("Root is", x)