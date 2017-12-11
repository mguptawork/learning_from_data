#!/usr/bin/python3
'''Module to solve problem 7 in hw 1 of learning from data'''
import random
import collections
import functools

Point = collections.namedtuple("Point", "x y")

sign = lambda x: (1, -1)[x < 0]

def create_target_function():
    '''Create a function which classifies points according to the ideal target function'''
    point_1 = create_random_points(1)[0]
    point_2 = create_random_points(1)[0]

    w_y = point_1.x - point_2.x
    w_x = point_2.y - point_1.y
    w_0 = point_1.y * (point_1.x +point_2.x) - point_1.x * (point_1.y + point_2.y)

    soln_parameters = (w_y, w_x, w_0)

    def target_function(point):
        '''target function'''
        return apply_perceptron(soln_parameters, point)

    return target_function, soln_parameters

def create_random_points(count):
    '''random points in a list of length count'''
    return [Point(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(count)]

def create_known_data_points():
    '''data_points with solution list and also the parameters of the soln'''
    target_function, soln_parameters = create_target_function()
    known_points = create_random_points(1000)
    known_data_points = list(zip(known_points, map(target_function, known_points)))
    return known_data_points, soln_parameters

def apply_perceptron(parameters, point):
    w_y, w_x, w_0 = parameters
    return sign(point.y * w_y + point.x * w_x + w_0)

def normalize_parameters(parms):
    '''
    normalize parameters
    '''
    w_y, w_x, w_0 = parameters
    determinant = w_y*w_y + w_x*w_x + w_0*w_0
    if determinant == 0:
        return (parms)
    else:
        return (w_y/determinant, w_x/determinant, w_0/determinant)

def main():
    '''a function that creates goal points and then finds a match'''
    known_data_points, soln_parameters = create_known_data_points()
    # print(known_data_points, soln_parameters)

    parms = [0,0,0]
    count = 0
    # unsolved_points_solns = known_data_points
    perceptron = functools.partial(apply_perceptron, parms)
    unsolved_points_solns = [point_soln for point_soln in known_data_points if perceptron(point_soln[0]) != point_soln[1]] 
    while len(unsolved_points_solns) > 0:
        random_unsolved_point = random.choice(unsolved_points_solns)
        parms[0] = parms[0] + random_unsolved_point[1] * random_unsolved_point[0].y
        parms[1] = parms[1] + random_unsolved_point[1] * random_unsolved_point[0].x
        parms[2] = parms[2] + random_unsolved_point[1]
        perceptron = functools.partial(apply_perceptron, parms)
        unsolved_points_solns = [point_soln for point_soln in known_data_points if perceptron(point_soln[0]) != point_soln[1]]
        count += 1

    print(count)
    print(parms)
    print(soln_parameters)

if __name__ == '__main__':
    main()
