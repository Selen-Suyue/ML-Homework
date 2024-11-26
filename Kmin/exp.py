import math
import random
from termcolor import cprint
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# add this column to prevent from exceeding python's recursive restirct
sys.setrecursionlimit(1000000)


def select_kth_smallest(A, low, high, k):
    global iteration_count
    iteration_count += 1

    p = high - low + 1
    if p < 44:
        A[low:high + 1] = sorted(A[low:high + 1])
        return A[low:low + k] if k <= p else A
    
    #assigin group
    q = math.floor(p / 5)
    M = []
    for i in range(q):
        start = low + i * 5
        end = min(low + (i + 1) * 5, high + 1)
        A[start:end] = sorted(A[start:end])
        M.append(A[start + 2])
    
    #recursive
    mm = select_kth_smallest(M, 0, len(M) - 1, math.ceil(len(M) / 2))[-1]
    A1 = []
    A2 = []
    A3 = []
    for a in A[low:high + 1]:
        if a < mm:
            A1.append(a)
        elif a == mm:
            A2.append(a)
        else:
            A3.append(a)
    
    #divede the boundary
    if len(A1) >= k:
        return select_kth_smallest(A1, 0, len(A1) - 1, k)
    elif len(A1) + len(A2) >= k:
        return A1 + A2[:k - len(A1)]
    else:
        return A1 + A2 + select_kth_smallest(A3, 0,
                                             len(A3) - 1,
                                             k - len(A1) - len(A2))


def select_kth_smallest_adaptive(A, low, high, k):
    global iteration_count_adapt
    iteration_count_adapt += 1

    p = high - low + 1
    if p < 44:
        A[low:high + 1] = sorted(A[low:high + 1])
        return A[low:low + k] if k <= p else A

    ratio = k / p
    
    # map ratio to index
    if ratio <= 0.2:
        pivot_index = 0
    elif ratio <= 0.4:
        pivot_index = 1
    elif ratio <= 0.6:
        pivot_index = 2
    elif ratio <= 0.8:
        pivot_index = 3
    else:
        pivot_index = 4

    q = math.floor(p / 5)
    M = []
    for i in range(q):
        start = low + i * 5
        end = min(low + (i + 1) * 5, high + 1)
        A[start:end] = sorted(A[start:end])
        M.append(A[start + pivot_index])

    mm = select_kth_smallest_adaptive(
        M, 0,
        len(M) - 1,
        math.ceil(len(M) / 5) * (pivot_index + 1))[-1]

    A1 = []
    A2 = []
    A3 = []
    for a in A[low:high + 1]:
        if a < mm:
            A1.append(a)
        elif a == mm:
            A2.append(a)
        else:
            A3.append(a)

    if len(A1) >= k:
        return select_kth_smallest_adaptive(A1, 0, len(A1) - 1, k)
    elif len(A1) + len(A2) >= k:
        return A1 + A2[:k - len(A1)]
    else:
        return A1 + A2 + select_kth_smallest_adaptive(A3, 0,
                                                      len(A3) - 1,
                                                      k - len(A1) - len(A2))


if __name__ == "__main__":
    '''
     args for exp, you can modify the arraylenth and k_percentages 
     the k_percentage means the k-th's ratio in the whole array
    '''
    array_lengths = [200, 300, 400]
    # array_lengths = [10**2,10**3,10**4]
    k_percentages = range(1, 10)
    # k_percentages = range(1, 101)

    for array_length in array_lengths:
        results = {"original": [], "adaptive": []}
        for k_percentage in tqdm(k_percentages,
                                 desc=f"Array Length: {array_length}"):

            #convert percentage to factual k-th
            k = int(array_length * k_percentage / 100)
            iterations_original = []
            iterations_adaptive = []
            for _ in range(10):

                # the binary method
                A = random.sample(range(1, 100000000), array_length)
                global iteration_count
                iteration_count = 0
                select_kth_smallest(A.copy(), 0, len(A) - 1, k)
                iterations_original.append(iteration_count)
                
                # You can validate whether the algorithm's result is correct here
                # k_smallest_elements = select_kth_smallest(A, 0, len(A) - 1, k)
                # cprint(f"The {k} smallest elements are: {k_smallest_elements}", "green")   
                  
                ## the ratio allocate method
                global iteration_count_adapt
                iteration_count_adapt = 0
                select_kth_smallest_adaptive(A.copy(), 0, len(A) - 1, k)
                iterations_adaptive.append(iteration_count_adapt)

            results["original"].append(
                sum(iterations_original) / len(iterations_original))
            results["adaptive"].append(
                sum(iterations_adaptive) / len(iterations_adaptive))

        #visuialize the results
        plt.figure()
        plt.plot(k_percentages, results["original"], label="Original")
        plt.plot(k_percentages, results["adaptive"], label="Adaptive")
        plt.xlabel("k Percentage")
        plt.ylabel("Average Iterations")
        plt.title(f"Array Length: {array_length}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Output1/results_littleratio_{array_length}.png")
        # plt.savefig(f"Output/results_{array_length}.png")
        plt.show()
