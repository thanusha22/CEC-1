import numpy
import math

from numpy import dot, ones, array, ceil
from opfunu.cec.cec2014.utils import *

SUPPORT_DIMENSION = [2, 10, 20, 30, 50, 100]
SUPPORT_DIMENSION_2 = [10, 20, 30, 50, 100]



def F5(solution=None, name="Shifted and Rotated Ackley’s Function", shift_data_file="shift_data_5.txt", bias=500):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2015 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_5_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2015 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f5_ackley__(z) + bias

def F6(solution=None, name="Shifted Rastrigin’s Function", shift_data_file="shift_data_8.txt", bias=800):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_8_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5.12 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f8_rastrigin__(z) + bias

def F7(solution=None, name="Shifted and Rotated Griewank’s Function", shift_data_file="shift_data_7.txt", bias=700):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_7_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 600 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f7_griewank__(z) + bias


def F10(solution=None, name="Shifted Schwefel’s Function", shift_data_file="shift_data_10.txt", bias=1000):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_10_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias

def F11(solution=None, name="Shifted and Rotated Schwefel’s Function", shift_data_file="shift_data_11.txt", bias=1100):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_11_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias

def F12(solution=None, name="Shifted and Rotated Katsuura Function", shift_data_file="shift_data_12.txt", bias=1200):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_12_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f10_katsuura__(z) + bias

def F13(solution=None, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function", shift_data_file="shift_data_15.txt", bias=1500):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_15_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return f13_expanded_griewank__(z) + bias

def F14(solution=None, name="Shifted and Rotated Expanded Scaffer’s F6 Function", shift_data_file="shift_data_16.txt", bias=1600):
    problem_size = len(solution)
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION:
        f_matrix = "M_16_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix) + 1
    return f14_expanded_scaffer__(z) + bias




def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -500, 500, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -5, 5, 4],
        "F14": ["F14", -5, 5, 2],
        "F15": ["F15", -32, 32, 30],
        "F16": ["F16", -600, 600, 30],
        "F17": ["F17", -50, 50, 30],
        
        
    }
    return param.get(a, "nothing")

