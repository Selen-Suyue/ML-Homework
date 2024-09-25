import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from termcolor import cprint
import matplotlib
import argparse

#e.g.  python Radix.py --low 1 --up 10000 --leng 200

def radix_sort(arr):
    frames = []
    m = max(arr)
    exp = 1
    while m // exp > 0:
        buckets = [[] for _ in range(10)]
        for i in arr:
            buckets[(i // exp) % 10].append(i)
        arr = [i for bucket in buckets for i in bucket]
        frames.append((arr.copy(), exp))  
        exp *= 10
    cprint("Sorted array:", "cyan")
    cprint(frames[-1][0], "cyan") 
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Radix Sort Visualizer')
    parser.add_argument('--low', type=int, default=1, help='Lower bound for random numbers')
    parser.add_argument('--up', type=int, default=9000, help='Upper bound for random numbers')
    parser.add_argument('--leng', type=int, default=40, help='Length of the array')

    args = parser.parse_args()
    arr = random.sample(range(args.low, args.up), args.leng) 
    cprint("Original array:", "light_magenta")
    cprint(arr, "light_magenta")  
    with open('input_result.txt', 'a') as f:
        f.write(f"Original array: {arr}\n")
        f.write(f"Sorted array: {radix_sort(arr)[-1][0]}\n\n") 