#!/usr/bin/env python3
def poly_derivative(poly):
    if type(poly) is not list or poly == []:
        return None
    elif len(poly) < 2:
        return[0]
    else:
        derivative = poly.copy()
        exponent = 1
        derivative.pop(0)
        for i in range(len(derivative)):
            derivative[i] = derivative [i] * exponent
            exponent += 1
        return derivative 
