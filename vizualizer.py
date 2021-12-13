import copy
import math
import random

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy
import numpy as np
from matplotlib import animation
from numpy import e


def anim_func(n, x, y, z, point):
    point.set_data_3d(numpy.array([x[n], y[n], z[n]]))
    point.set_3d_properties(z[n], 'z')
    return point


def draw(lower, upper, function, ax):
    x, y = numpy.meshgrid(numpy.linspace(lower, upper), numpy.linspace(lower, upper))
    z = function([x, y])
    ax.plot_surface(x, y, z, cmap='jet', alpha=0.2)


def new_best(x, y, z):
    print("Got new best at X: %f Y: %f Z: %f" % (x, y, z))


def diff_ev_mutate(x, mut_constant):
    return numpy.array(x[0]) + mut_constant * (numpy.array(x[1]) - numpy.array(x[2]))


def get_pb(swarm, function):
    personal_best = function(swarm[0])
    personal_best_index = 0
    for i, particle in enumerate(swarm):
        particle_value = function(particle)
        if personal_best > particle_value:
            personal_best_index = i
            personal_best = particle_value

    return swarm[personal_best_index]


def animate_soma(i, best_xxs, best_yys, best_zzs, points):
    for j in range(len(best_xxs[0])):
        x = best_xxs[i][j]
        y = best_yys[i][j]
        z = best_zzs[i][j]

        points[j].set_data_3d(np.array([x, y, z]))
        points[j].set_3d_properties(z, 'z')
    return points


def get_leader(swarm, function):
    personal_best = function(swarm[0])
    personal_best_index = 0
    for i, particle in enumerate(swarm):
        particle_value = function(particle)
        if personal_best > particle_value:
            personal_best_index = i
            personal_best = particle_value

    return swarm[personal_best_index]


def generate_city_positions(city_list):
    x = []
    y = []

    for city in city_list:
        x.append(city.x)
        y.append(city.y)
    return x, y


def generate_pop(city_list, total_population):
    populations = []
    for i in range(0, total_population):
        tmp_gen = []
        while len(tmp_gen) != len(city_list):
            random_index = random.randrange(0, len(city_list))
            if city_list[random_index] not in tmp_gen:
                tmp_gen.append(city_list[random_index])

        populations.append(tmp_gen)
    return populations


def mutate(populationM):
    index_a = random.randrange(0, len(populationM))
    index_b = random.randrange(0, len(populationM))
    tmp = populationM[index_a]
    populationM[index_a] = populationM[index_b]
    populationM[index_b] = tmp
    return populationM


def calculate_total_distance(track):
    total_distance = 0
    for i in range(0, len(track) - 1):
        total_distance += track[i].get_distance(track[i + 1])
    total_distance += track[len(track) - 1].get_distance(track[0])
    return total_distance


def cross_breed(population_count):
    new_populations = []
    fitness_populations = []

    for pop in population_count:
        fitness_populations.append((pop, calculate_total_distance(pop)))

    fitness_populations.sort(key=lambda x: x[1], reverse=False)

    for pop in population_count:
        next_population = fitness_populations[random.randrange(0, 5)][0]
        new_population_start = []
        new_population_middle = []
        new_population_end = []
        start = int(random.random() * len(pop) - 1)
        stop = int(random.random() * len(pop))
        while stop == start:
            stop = int(random.random() * len(pop))
        if stop < start:
            tmp = start
            start = stop
            stop = tmp

        for i in range(0, len(pop)):
            if i >= stop:
                new_population_end.append(pop[i])
            if i <= start:
                new_population_start.append(pop[i])

        while len(new_population_start + new_population_middle + new_population_end) != len(pop):
            indexer = 0
            while True:
                if (next_population[indexer] not in new_population_start) and (
                        next_population[indexer] not in new_population_middle) and (
                        next_population[indexer] not in new_population_end):
                    new_population_middle.append(next_population[indexer])
                    break
                else:
                    indexer += 1
        final_population = new_population_start + new_population_middle + new_population_end

        if random.random() < 0.5:
            new_populations.append(mutate(final_population))
        else:
            new_populations.append(final_population)
    return new_populations


def find_best(population):
    best_population = population[0]
    best_distance = calculate_total_distance(population[0])

    for pop in population:
        if calculate_total_distance(pop) < best_distance:
            best_distance = calculate_total_distance(pop)
            best_population = pop
    return best_population


class Vizualizer:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.parameters = numpy.zeros(self.dimension)
        self.function = numpy.inf

    class City:
        def __init__(self, x, y, id):
            self.id = id
            self.x = x
            self.y = y

        def get_distance(self, city):
            return math.sqrt(math.pow(city.x - self.x, 2) + math.pow(city.y - self.y, 2))

    def hill_climb(self, functions, neighbour_count=10, sigma=0.5, total_iterations=1000):
        for func in functions:
            print("Starting %s" % func.__name__)
            self.hc(func, neighbour_count, sigma, total_iterations)
            print("-------")

    def simulated_annealing(self, functions, initial_temperature=250, minimal_temperature=0.1, cooling_coefficient=0.95,
                            sigma=0.5):
        for func in functions:
            print("Starting %s" % func.__name__)
            self.sa(func, initial_temperature, minimal_temperature, cooling_coefficient, sigma)
            print("-------")

    def differential_evolution(self, functions, parents_count=5, iteration_count=10, mut_constant=0.5,
                               crossover_ran=0.5):
        for func in functions:
            print("Starting %s" % func.__name__)
            self.diff(func, parents_count, iteration_count, mut_constant, crossover_ran)
            print("-------")

    def self_organizing_algorithm_all_to_one(self, functions, pop_size=20, prt=0.4, path_len=3.0, step=0.11, m_max=100):
        for func in functions:
            print("Starting %s" % func.__name__)
            self.soma(func, pop_size, prt, path_len, step, m_max)
            print("-------")

    def particle_swarm(self, functions, pop_size=20, migration_cycles=50, c1=2, c2=2, v_mini=-1, v_maxi=1):
        for func in functions:
            print("Starting %s" % func.__name__)
            self.ps(func, pop_size, migration_cycles, c1, c2, v_mini, v_maxi)
            print("-------")

    def hc(self, function, neighbour_count, sigma, total_iterations):
        points_to_show = []
        best_point = [self.upper_bound, self.upper_bound]
        z = function(best_point)
        best_point.append(z)
        for i in range(total_iterations):
            points_to_show.append(best_point)
            neighbours = self.get_neighbours_gauss([best_point[0], best_point[1]], neighbour_count, sigma)
            for neighbour in neighbours:
                z = function(neighbour)
                if z < best_point[2]:
                    new_best(neighbour[0], neighbour[1], z)
                    best_point = [neighbour[0], neighbour[1], z]
        self.show(points_to_show, function)
        return

    def sa(self, function, initial_temperature, minimal_temperature, cooling_coefficient, sigma):
        temperature = initial_temperature
        points_to_show = []
        best_point = [self.upper_bound, self.upper_bound]
        z = function(best_point)
        best_point.append(z)
        points_to_show.append(best_point)
        worse_accepted = 0
        worse_refused = 0
        while temperature > minimal_temperature:
            neighbour = self.get_neighbours_gauss([best_point[0], best_point[1]], 1, sigma)[0]
            z = function(neighbour)
            if z < best_point[2]:
                best_point = [neighbour[0], neighbour[1], z]
                new_best(neighbour[0], neighbour[1], z)
                points_to_show.append(best_point)
            else:
                r = numpy.random.uniform()
                z_diff = z - best_point[2]
                if r < e ** (-(z_diff / temperature)):
                    worse_accepted += 1
                    best_point = [neighbour[0], neighbour[1], z]
                    points_to_show.append(best_point)
                else:
                    worse_refused += 1
            temperature = temperature * cooling_coefficient
        self.show(points_to_show, function)
        return best_point, False

    def diff(self, function, parents_count, iteration_count, mut_constant, crossover_ran):
        pop = []
        points_to_show = []
        for i in range(parents_count):
            parent = self.get_random_solution(function)
            pop.append(parent)
        pop.sort(key=lambda a: a[2])
        points_to_show.append(pop[0])

        for i in range(iteration_count):
            new_pop = copy.deepcopy(pop)
            for (index, item) in enumerate(pop):
                r1 = numpy.random.randint(0, parents_count)
                while r1 == index:
                    r1 = numpy.random.randint(0, parents_count)
                r2 = numpy.random.randint(0, parents_count)
                while r2 in [index, r1]:
                    r2 = numpy.random.randint(0, parents_count)
                r3 = numpy.random.randint(0, parents_count)
                while r3 in [index, r1, r2]:
                    r3 = numpy.random.randint(0, parents_count)
                mutated = diff_ev_mutate([pop[r1], pop[r2], pop[r3]], mut_constant)
                for j in range(self.dimension):
                    if mutated[j] < self.lower_bound:
                        mutated[j] = self.lower_bound
                    elif mutated[j] > self.upper_bound:
                        mutated[j] = self.upper_bound
                trial = self.diff_ev_crossover(mutated, item, crossover_ran, function)
                if trial[self.dimension] <= item[self.dimension]:
                    new_pop[index] = trial
            pop = new_pop
            best = None
            for item in pop:
                if best is None or item[self.dimension] < best[self.dimension]:
                    best = item
                    new_best(best[0], best[1], best[2])
            points_to_show.append(best)
        self.show(points_to_show, function)

    def soma(self, function, pop_size, prt, path_len, step, m_max):
        population = self.get_neighbours_gauss([self.upper_bound, self.upper_bound], pop_size, 0.5)
        leader = get_leader(population, function)
        population_solution = []
        m = 0
        while m < m_max:
            print(m)
            leader = get_leader(population, function)
            for i in range(len(population)):
                population[i] = self.recalculate_individual(function, population[i], leader, path_len, prt, step)

            population_solution.append(copy.deepcopy(population))
            m += 1
        if self.dimension == 2:
            self.animate_soma_sol(population_solution, function)
        return leader

    def ps(self, function, pop_size, migration_cycles, c1, c2, v_mini, v_maxi):
        swarm = self.get_neighbours_gauss([self.upper_bound, self.upper_bound], pop_size, 0.5)
        g_best = get_pb(swarm, function)
        p_best = copy.deepcopy(swarm)
        velocity = self.generate_swarm_velocity(pop_size, v_mini, v_maxi)
        swarm_solution = []
        current_cycle = 0

        while current_cycle < migration_cycles:
            for i in range(len(swarm)):
                velocity[i] = self.recalculate_particle_velocity(velocity[i], swarm[i], p_best[i], g_best,
                                                                 current_cycle, migration_cycles, c1)
                swarm[i] = self.fix_particle_bounds(np.add(swarm[i], velocity[i]))

                if function(swarm[i]) < function(p_best[i]):
                    p_best[i] = copy.deepcopy(swarm[i])
                    if function(p_best[i]) < function(g_best):
                        g_best = copy.deepcopy(p_best[i])

            swarm_solution.append(copy.deepcopy(swarm))
            c2 = c2
            current_cycle += 1

        if self.dimension == 2:
            self.animate_soma_sol(swarm_solution, function)
        return g_best

    def genetic_algorithm(self, city_count=10, total_populations=150):
        city_list = self.generate_cities(city_count, 200)
        population = generate_pop(city_list, total_populations)
        city_x, city_y = generate_city_positions(city_list)
        best_distance = math.inf
        plt.ion()

        while True:
            population = cross_breed(population)
            new_best_pop = find_best(population)
            new_best_distance = calculate_total_distance(new_best_pop)

            if new_best_distance < best_distance:
                best_pop = new_best_pop[:]
                best_distance = new_best_distance

                for aa in range(0, len(best_pop) - 1):
                    plt.plot([best_pop[aa].x, best_pop[aa + 1].x], [best_pop[aa].y, best_pop[aa + 1].y])
                plt.plot([best_pop[len(best_pop) - 1].x, best_pop[0].x], [best_pop[len(best_pop) - 1].y, best_pop[0].y])

                plt.scatter(city_x, city_y, s=60, c=(0, 0, 0), alpha=0.5)
                plt.title('TSP Current:' + str(best_distance))

                plt.draw()
                plt.pause(0.5)
                plt.clf()

    def generate_cities(self, count, boundaries):
        cities = []
        for i in range(0, count):
            new_x = random.randrange(0, boundaries)
            new_y = random.randrange(0, boundaries)
            cities.append(self.City(new_x, new_y, i))
        return cities

    def generate_swarm_velocity(self, pop_size, v_mini, v_maxi):
        p = []
        for xi in range(pop_size):
            pi = []
            for i in range(self.dimension):
                pi.append(np.random.uniform(v_mini, v_maxi))
            p.append(pi)
        return p

    def recalculate_particle_velocity(self, velocity, particle, p_best, g_best, i, migration_cycles, c1):
        ws = 0.9
        we = 0.4
        r1 = np.random.uniform()
        w = ws * ((ws - we) * i) / migration_cycles
        new_velocity = np.add(np.add(np.multiply(velocity, w), np.multiply((r1 * c1), (np.subtract(p_best, particle)))),
                              np.multiply((r1 * c1), (np.subtract(g_best, particle))))
        self.fix_particle_bounds(new_velocity)
        return new_velocity

    def fix_particle_bounds(self, velocity):
        for i in range(len(velocity)):
            if velocity[i] < self.lower_bound:
                velocity[i] = self.lower_bound
            elif velocity[i] > self.upper_bound:
                velocity[i] = self.upper_bound
        return velocity

    def correct_bounds(self, params):
        for i in range(params):
            if i == self.dimension:
                break
            if params[i] < self.lower_bound:
                params[i] = self.lower_bound
            elif params[i] > self.upper_bound:
                params[i] = self.upper_bound

    def show(self, points, function):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x, y, z = [], [], []
        draw(self.lower_bound, self.upper_bound, function, ax)
        for i in range(len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
            z.append(points[i][2])
        point, = ax.plot(xs=x, ys=y, zs=z, zdir='^')
        plt.suptitle(function.__name__)
        tmp = animation.FuncAnimation(fig, anim_func, len(x), interval=30, fargs=(x, y, z, point))
        plt.show()

    def get_neighbours_gauss(self, params, neighbour_count, sigma):
        neighbours = []
        for i in range(neighbour_count):
            neighbour = []
            for param in params:
                valid = False
                while not valid:
                    value = numpy.random.normal(param, sigma)
                    if self.lower_bound <= value <= self.upper_bound:
                        neighbour.append(value)
                        valid = True
            neighbours.append(neighbour)
        return neighbours

    def get_random_solution(self, function=None, offset=0):
        result = numpy.random.uniform(self.lower_bound + offset, self.upper_bound - offset, self.dimension).tolist()
        if function:
            result.append(function(result))
        return result

    def diff_ev_crossover(self, mutated, target, cr, function):
        p = self.get_random_solution()
        trial_vector = []
        rnd = numpy.random.randint(0, self.dimension)
        for i in range(len(p)):
            if p[i] < cr or i == rnd:
                trial_vector.append(mutated[i])
            else:
                trial_vector.append(target[i])
        trial_vector.append(function(trial_vector))
        return trial_vector

    def animate_soma_sol(self, best_solutions, function):
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        best_xxs = []
        best_yys = []
        best_zzs = []
        points = []

        for best_solution in best_solutions:
            best_xs = []
            best_ys = []
            best_zs = []
            for i in range(len(best_solution)):
                best_xs.append(best_solution[i][0])
                best_ys.append(best_solution[i][1])
                best_zs.append(function([best_solution[i][0], best_solution[i][1]]))
            best_xxs.append(best_xs)
            best_yys.append(best_ys)
            best_zzs.append(best_zs)

        draw(self.lower_bound, self.upper_bound, function, ax)
        for i in range(len(best_xxs[0])):
            point, = ax.plot([best_xxs[i][0]], [best_yys[i][0]], [best_zzs[i][0]], 'o')
            points.append(point)
            plt.suptitle(function.__name__)
        tmp = animation.FuncAnimation(fig, animate_soma, len(best_xxs), fargs=(best_xxs, best_yys, best_zzs, points),
                                      interval=50, repeat=False)
        plt.show()

    def recalculate_individual(self, function, individual, leader, path_length, prt, step):
        t = 0
        old_individual = copy.deepcopy(individual)
        new_individual = copy.deepcopy(individual)
        partial_individual = copy.deepcopy(individual)
        while t < path_length:
            for j in range(self.dimension):
                rnd = np.random.uniform(0, 1)
                prt_vector = 0
                if rnd < prt:
                    prt_vector = 1

                new_individual[j] = np.add(old_individual[j],
                                           np.multiply(np.subtract(leader[j], old_individual[j]), (t * prt_vector)))

            new_individual = self.fix_boundaries(new_individual)

            if function(new_individual) < function(partial_individual):
                partial_individual = copy.deepcopy(new_individual)
            t += step

        if function(partial_individual) < function(old_individual):
            return partial_individual

        return old_individual

    def fix_boundaries(self, individual):
        for j in range(self.dimension):
            if individual[j] < self.lower_bound:
                individual[j] = self.lower_bound
            elif individual[j] > self.upper_bound:
                individual[j] = self.upper_bound
        return individual
