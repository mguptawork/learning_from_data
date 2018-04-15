#!/home/mayank/anaconda3/bin/python3
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
    two_points = create_random_df(2)

    w_0 = two_points['x2'][0] *  two_points['x1'][1] - two_points['x1'][0] * two_points['x2'][1]
    w_1 = two_points['x2'][1] - two_points['x2'][0]
    w_2 = two_points['x1'][0] - two_points['x1'][1]
    soln_parameters = pandas.Series([w_0, w_1, w_2])

    # def target_function(point):
    #     '''target function'''
    #     return apply_perceptron(soln_parameters, point)

    return soln_parameters

def create_random_df(count):
    '''random panda series valid for 2 dimensions'''
    number_of_dimensions = 2
    columns = list(map(lambda num: 'x'+str(num + 1), range(number_of_dimensions)))  # ['x1','x2']
    return pandas.DataFrame(numpy.random.uniform(-1,1,(count,number_of_dimensions)), columns=columns)

def apply_parms(df, parms):
    return list(map(sign, numpy.dot(df, parms)))

def create_known_data_points(number_of_points):
    '''data_points with solution list and also the parameters of the soln'''
    soln_parameters = create_target_function()
    known_points = create_random_df(number_of_points)
    known_points.insert(0,'x0',1)       #constant factor

    # print(known_points.shape)
    # print(known_points)
    # print(soln_parameters.shape)

    #### Do not work because x0,x1, x2 do not match index 0,1,2
    # print(known_points.dot(soln_parameters))
    # known_points['soln'] = soln_parameters.dot(known_points.T)
    # soln = soln_parameters.T.dot(known_points.T)
    #######################################

    known_points['soln'] = list(map(sign, numpy.dot(known_points, soln_parameters)))

    return known_points, soln_parameters

def get_input_points(df):
    return df.filter(regex='^x',axis=1)

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

    # print(list(get_input_points(known_data_points).columns))
    parms = pandas.Series(0, index=get_input_points(known_data_points).columns)
    # print(parms)
    count = 0
    known_data_points['partial_soln'] = apply_parms(get_input_points(known_data_points), parms)

    candidate_df = known_data_points[known_data_points['soln']!= known_data_points['partial_soln']]
    # print(candidate_df)
    # print()

    while len(candidate_df) > 0:
        random_unsolved_point = candidate_df.sample(axis=0)
        # print(random_unsolved_point)
        parms = parms + random_unsolved_point['soln'].iloc[0]*get_input_points(random_unsolved_point).iloc[0]
        # print(parms)
        # print(type(parms))
        known_data_points['partial_soln'] = apply_parms(get_input_points(known_data_points), parms)
        candidate_df = known_data_points[known_data_points['soln']!= known_data_points['partial_soln']]
        count += 1
        # print(candidate_df)
        # break


    # print(count)
    # print(normalize_parameters(parms))
    # print(normalize_parameters(soln_parameters))
    # print(calculate_accuracy(soln_parameters,parms,1000))
    # print("soln_parameters")
    # print_points(soln_parameters, known_data_points)
    # print("pla solution")
    # print_points(parms, known_data_points)
    
    # print('known_data_points:\n', known_data_points, '\n')
    print('soln_parameters:\n', soln_parameters, '\n')
    print('parms:\n', parms, '\n')
    # print(apply_parms(get_input_points(known_data_points), parms))

if __name__ == '__main__':
    main()
