#!/usr/bin/python3
'''Module to solve problem 7 in hw 1 of learning from data'''
import random
import collections
import functools
import pandas
import numpy

Point = collections.namedtuple("Point", "x y")

sign = lambda x: (1, -1)[x < 0]

def create_target_function():
    '''Create a function which classifies points according to the ideal target function'''
    series1 = create_random_df(1)
    series2 = create_random_df(1)

    w_y = series1.values[0][1] - series2.values[0][1]
    w_x = series2.values[0][2] - series1.values[0][2]
    w_0 = series1.values[0][2] * (series1.values[0][1] +series2.values[0][1]) - series1.values[0][1] * (series1.values[0][2] + series2.values[0][2])

    soln_parameters = pandas.Series([w_0, w_x, w_y])

    def target_function(point):
        '''target function'''
        return apply_perceptron(soln_parameters, point)

    return target_function, soln_parameters

def create_random_df(count):
    '''random panda series valid for 2 dimensions'''
    return pandas.DataFrame([pandas.Series([1, random.uniform(-1, 1), random.uniform(-1, 1)]) for _ in range(count)])

def create_random_points(count):
    '''random points in a list of length count'''
    return [Point(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(count)]

def create_known_data_points(number_of_points):
    '''data_points with solution list and also the parameters of the soln'''
    target_function, soln_parameters = create_target_function()
    known_points = create_random_df(number_of_points)
    known_points['soln'] = map(target_function, known_points)
    return known_points, soln_parameters

def apply_perceptron(parameters, point):
    return sign(parameters.dot(point))

def normalize_parameters(parms):
    '''
    normalize parameters
    '''
    w_y, w_x, w_0 = parms
    determinant = w_y*w_y + w_x*w_x + w_0*w_0
    if determinant == 0:
        return (parms)
    else:
        return (w_y/determinant, w_x/determinant, w_0/determinant)

def print_points(parms, known_data_points):
    '''
    print points and target match
    '''
    perceptron = functools.partial(apply_perceptron, parms)
    for point, target in known_data_points:
        print(point, target, perceptron(point))

def calculate_accuracy(soln_parms, guess_parms, number_of_points=1000):
    '''
    calculate accuracy of soln vs pla guess
    '''
    known_perceptron = functools.partial(apply_perceptron, soln_parms)
    guess_perceptron = functools.partial(apply_perceptron, guess_parms)
    random_points = create_random_df(number_of_points)
    incorrect = 0
    for point in random_points:
        if known_perceptron(point)!= guess_perceptron(point):
            incorrect += 1
    return incorrect/float(number_of_points)

def main():
    '''a function that creates goal points and then finds a match'''
    known_data_points, soln_parameters = create_known_data_points(100)
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
    print(normalize_parameters(parms))
    print(normalize_parameters(soln_parameters))
    print(calculate_accuracy(soln_parameters,parms,1000))
    # print("soln_parameters")
    # print_points(soln_parameters, known_data_points)
    # print("pla solution")
    # print_points(parms, known_data_points)


if __name__ == '__main__':
    main()
