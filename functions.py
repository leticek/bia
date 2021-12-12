import numpy as np
from numpy import pi, cos, exp, sqrt, e


def sphere(params):
    res = []
    for val in params:
        res.append(val ** 2)
    return sum(res)


def ackley(params=[]):
    a = 20
    b = 0.2
    c = 2 * pi
    d = 2
    sum_pow = 0
    sum_cos = 0
    for item in params:
        sum_pow += item ** 2
        sum_cos += cos(c * item)
    return -a * exp(-b * sqrt(1 / d * sum_pow)) - exp(1 / d * (sum_cos)) + a + e
