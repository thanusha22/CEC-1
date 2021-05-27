import numpy
import math

from numpy import dot, ones, array, ceil
from opfunu.cec.cec2014.utils import *

SUPPORT_DIMENSION = [2, 10, 20, 30, 50, 100]
SUPPORT_DIMENSION_2 = [10, 20, 30, 50, 100]


def C1(solution=None, name="Rotated Bent Cigar Function", f_shift_file="shift_data_1_D", f_matrix_file="M_1_D", bias=100):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f1_bent_cigar__(z) + bias


def C2(solution=None, name="Rotated Discus Function", f_shift_file="shift_data_2_D", f_matrix_file="M_2_D", bias=200):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix)
    return f2_discus__(z) + bias


def C3(solution=None, name="Shifted and Rotated Weierstrass Function", f_shift_file="shift_data_3_D", f_matrix_file="M_3_D", bias=300):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(0.5*(solution - shift_data) / 100, matrix)
    return f3_weierstrass__(z) + bias


def C4(solution=None, name="Shifted and Rotated Schwefel’s Function", f_shift_file="shift_data_4_D", f_matrix_file="M_4_D", bias=400):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(1000 * (solution - shift_data) / 100, matrix)
    return f4_modified_schwefel__(z) + bias


def C5(solution=None, name="Shifted and Rotated Ackley’s Function", shift_data_file="shift_data_5.txt", bias=500):
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



def C6(solution=None, name="Shifted Rastrigin’s Function", shift_data_file="shift_data_8.txt", bias=800):
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



def C7(solution=None, name="Shifted and Rotated Griewank’s Function", shift_data_file="shift_data_7.txt", bias=700):
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




def C8(solution=None, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function",
       f_shift_file="shift_data_8_D", f_matrix_file="M_8_D", bias=800):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(5 * (solution - shift_data) / 100, matrix) + 1
    return f8_expanded_griewank__(z) + bias


def C9(solution=None, name="Shifted and Rotated Expanded Scaffer’s F6 Function",
       f_shift_file="shift_data_9_D", f_matrix_file="M_9_D", bias=900):
    problem_size = len(solution)
    f_shift, f_matrix = check_problem_size(problem_size, f_shift_file, f_matrix_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    z = dot(solution - shift_data, matrix) + 1
    return f9_expanded_scaffer__(z) + bias


def C10(solution=None, name="Shifted Schwefel’s Function", shift_data_file="shift_data_10.txt", bias=1000):
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

def C11(solution=None, name="Shifted and Rotated Schwefel’s Function", shift_data_file="shift_data_11.txt", bias=1100):
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


def C12(solution=None, name="Shifted and Rotated Katsuura Function", shift_data_file="shift_data_12.txt", bias=1200):
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

def C13(solution=None, name="Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function", shift_data_file="shift_data_15.txt", bias=1500):
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

def C14(solution=None, name="Shifted and Rotated Expanded Scaffer’s F6 Function", shift_data_file="shift_data_16.txt", bias=1600):
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



def C15(solution=None, name="Hybrid Function 1 (N=3)", f_shift_file="shift_data_10_D", f_matrix_file="M_10_D",
        f_shuffle_file="shuffle_data_10_D", bias=1000):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):]
    mz = dot(solution - shift_data, matrix)
    return f4_modified_schwefel__(mz[idx1]) + f12_rastrigin__(mz[idx2]) + f13_elliptic__(mz[idx3]) + bias



def C16(solution=None, name="Hybrid Function 2 (N=4)", f_shift_file="shift_data_11_D", f_matrix_file="M_11_D",
        f_shuffle_file="shuffle_data_11_D", bias=1100):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))

    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1+n2+n3)]
    idx4 = shuffle[(n1 + n2 + n3):]
    mz = dot(solution - shift_data, matrix)
    return f11_griewank__(mz[idx1]) + f3_weierstrass__(mz[idx2]) + f10_rosenbrock__(mz[idx3]) + f9_expanded_scaffer__(mz[idx4]) + bias


def C17(solution=None, name="Hybrid Function 3 (N=5)", f_shift_file="shift_data_12_D", f_matrix_file="M_12_D",
        f_shuffle_file="shuffle_data_12_D", bias=1200):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))

    f_shift, f_matrix, f_shuffle = check_problem_size(problem_size, f_shift_file, f_matrix_file, f_shuffle_file)
    shift_data = load_shift_data__(f_shift)
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle) - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1 + n2 + n3):(n1+n2+n3+n4)]
    idx5 = shuffle[(n1+n2+n3+n4):]
    mz = dot(solution - shift_data, matrix)
    return f5_katsuura__(mz[idx1]) + f6_happy_cat__(mz[idx2]) + f8_expanded_griewank__(mz[idx3]) + f4_modified_schwefel__(mz[idx4]) + \
           f14_ackley__(mz[idx5]) + bias







def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "C1": ["C1", -100, 100, 30],
        "C2": ["C2", -10, 10, 30],
        "C3": ["C3", -100, 100, 30],
        "C4": ["C4", -100, 100, 30],
        "C5": ["C5", -30, 30, 30],
        "C6": ["C6", -500, 500, 30],
        "C7": ["C7", -1.28, 1.28, 30],
        "C8": ["C8", -500, 500, 30],
        "C9": ["C9", -5.12, 5.12, 30],
        "C10": ["C10", -32, 32, 30],
        "C11": ["C11", -600, 600, 30],
        "C12": ["C12", -50, 50, 30],
        "C13": ["C13", -5, 5, 4],
        "C14": ["C14", -5, 5, 2],
        "C15": ["C15", -32, 32, 30],
        "C16": ["C16", -600, 600, 30],
        "C17": ["C17", -50, 50, 30],
        
        
    }
    return param.get(a, "nothing")

