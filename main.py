from matplotlib import pyplot as plt
from numpy import linspace, arange, meshgrid

import vizualizer

from functions import *

functions = [sphere, schwefel, rosenbrock, rastrigin, griewank, levy, michalewicz, zakharov, ackley]


def main():
    vizual = vizualizer.Vizualizer(dimension=2, lower_bound=-10, upper_bound=10)
    # vizual.hill_climb(functions=functions, neighbour_count=10, sigma=0.5, total_iterations=1500)
    # vizual.simulated_annealing(functions, initial_temperature=250, minimal_temperature=0.1, cooling_coefficient=0.97,
    #                       sigma=0.5)
    # vizual.genetic_algorithm(city_count=15, total_populations=200)
    # vizual.differential_evolution(functions, parents_count=20, iteration_count=1000, mut_constant=0.5,
    #                             crossover_ran=0.5)
    # vizual.particle_swarm(functions, pop_size=20, migration_cycles=50, c1=2, c2=2, v_mini=-1, v_maxi=1)
    # vizual.self_organizing_algorithm_all_to_one(functions, pop_size=20, prt=0.4, path_len=3.0, step=0.11, m_max=100)
    # vizual.AntColony(total_ants=20, total_best=5, n_iterations=10000, decay=0.95, total_cities=20)


if __name__ == '__main__':
    main()
