import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy
from matplotlib import animation


def anim_func(n, x, y, z, point):
    point.set_data_3d(numpy.array([x[n], y[n], z[n]]))
    point.set_3d_properties(z[n], 'z')
    return point


def draw(lower, upper, function, ax):
    X, Y = numpy.meshgrid(numpy.linspace(lower, upper), numpy.linspace(lower, upper))
    Z = function([X, Y])
    ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.1)


class Vizualizer:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.parameters = numpy.zeros(self.dimension)
        self.function = numpy.inf

    def hill_climb(self, function, neighbour_count=10, sigma=0.5, total_iterations=1000):
        best_points = []
        best_point = [self.upper_bound, self.upper_bound]
        z = function(best_point)
        best_point.append(z)

        for i in range(total_iterations):
            print('-bp-| x: %f y: %f z: %f' % (best_point[0], best_point[1], best_point[2]))
            best_points.append(best_point)
            neighbours = self.get_neighbours_gauss([best_point[0], best_point[1]], neighbour_count, sigma)
            found_better_point = False

            for neighbour in neighbours:
                z = function(neighbour)
                print(' nb | x: %f y: %f z: %f' % (neighbour[0], neighbour[1], z))
                if z < best_point[2]:
                    found_better_point = True
                    best_point = [neighbour[0], neighbour[1], z]

        self.show(best_points, function)
        return

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
        x = []
        y = []
        z = []

        draw(self.lower_bound, self.upper_bound, function, ax)
        for i in range(len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
            z.append(points[i][2])

        point, = ax.plot(xs=x, ys=y, zs=z, zdir='^')
        tmp = animation.FuncAnimation(fig, anim_func, len(x), interval=50, fargs=(x, y, z, point))
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
