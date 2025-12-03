def func(x):

    return ( x**2 - 2 )

def numerical_differentiation(x, h = 0.0001):

    return ( func(x + h) - func(x - h) ) / (2 * h)

def newton(x, h = 0.0001):

    return x - ( func(x) / numerical_differentiation(x, h) )