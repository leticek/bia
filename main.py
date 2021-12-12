from matplotlib import pyplot as plt
from numpy import linspace, arange, meshgrid

import vizualizer

from functions import *

functions = [sphere, schwefel, rosenbrock, rastrigin, griewank, levy, michalewicz, zakharov, ackley]


def main():
    vizual = vizualizer.Vizualizer(dimension=2, lower_bound=-50, upper_bound=50)
    # vizual.hill_climb(functions=functions, neighbour_count=10, sigma=0.5, total_iterations=3000)
    # vizual.simulated_annealing(functions, initial_temperature=250, minimal_temperature=0.1, cooling_coefficient=0.97,
    #                           sigma=0.5)
    # vizual.differential_evolution(functions, parents_count=20, iteration_count=1000, mut_constant=0.5,
    #                             crossover_ran=0.5)
    # vizual.self_organizing_algorithm_all_to_one(functions, pop_size=20, prt=0.4, path_len=3.0, step=0.11, m_max=100)
    vizual.particle_swarm(functions, pop_size=20, migration_cycles=50, c1=2, v_mini=-1, v_maxi=1)


if __name__ == '__main__':
    main()
