import numpy as np
from numpy import pi, cos, exp, sqrt, e, sin


def sphere(params):
    res = []
    for val in params:
        res.append(val ** 2)
    return sum(res)


def schwefel(params):
    dimension = len(params)
    total = 0.0
    for i in params:
        total += i * sin(sqrt(abs(i)))
    return 418.9829 * dimension - total


def rosenbrock(params):
    return 100 * (params[1] - params[0] ** 2) ** 2 + (params[0] - 1) ** 2


def rastrigin(params):
    dimension = len(params)
    return 10 * dimension + (
            (params[0] ** 2 - 10 * cos(2 * pi * params[0])) + (params[1] ** 2 - 10 * cos(2 * pi * params[1])))


def griewank(params):
    sum1, sum2 = 0.0, 0.0
    for index, item in enumerate(params):
        sum1 += (item ** 2) / 4000
        sum2 *= cos(item / (sqrt(index + 1)))
    return sum1 - sum2 + 1


def levy(params):
    dimension = len(params)
    w_d = 1 + (params[dimension - 1] - 1) / 4
    total = 0.0
    for item in params:
        w = 1 + (item - 1) / 4
        tmp = ((w - 1) ** 2) * (1 + 10 * sin(pi * w + 1) ** 2) + (w_d - 1) ** 2 * (1 + sin(2 * pi * w_d))
        total += tmp
    return total


def michalewicz(params):
    m = 10
    dimension = len(params)
    total = 0.0
    for i in range(dimension):
        total -= (sin(params[i]) * (sin((i * params[i] ** 2) / pi) ** (2 * m)))
    return total


def zakharov(params):
    dimension = len(params)
    sum1, sum2, sum3, result = 0.0, 0.0, 0.0, 0.0
    for i in range(dimension):
        sum1 = params[i] ** 2
        sum2 = (0.5 * i * params[i]) ** 2
        sum3 = (0.5 * i * params[i]) ** 4
        result += sum1 + sum2 + sum3
    return result


def ackley(params):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = []
    sum2 = []

    for val in params:
        sum1.append(val ** 2)
        sum2.append(np.cos(c * val))
    part1 = -a * np.exp(-b * np.sqrt(sum(sum1) / len(params)))
    part2 = -np.exp(sum(sum2) / len(params))

    return a + np.exp(1) + part1 + part2
