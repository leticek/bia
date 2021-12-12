from matplotlib import pyplot as plt
from numpy import linspace, arange, meshgrid

import vizualizer

from functions import sphere, ackley


def main():
    solution = vizualizer.Vizualizer(2, -20, 20)
    solution.hill_climb(ackley, 8, 0.5)


if __name__ == '__main__':
    main()
