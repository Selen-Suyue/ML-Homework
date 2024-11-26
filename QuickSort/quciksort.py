import random
from termcolor import cprint
import argparse

# e.g.  python QuickSort.py --low 1 --up 10000 --leng 200

def partition(nums: list[int], left: int, right: int) -> int:
    """哨兵划分"""
    i, j = left, right
    while i < j:
        while i < j and nums[j] >= nums[left]:
            j -= 1  
        while i < j and nums[i] <= nums[left]:
            i += 1  
        nums[i], nums[j] = nums[j], nums[i]
    nums[i], nums[left] = nums[left], nums[i]
    return i  

def quick_sort(nums: list[int], left: int, right: int):
    """快速排序"""
    if left >= right:
        return
    pivot = partition(nums, left, right)
    
    quick_sort(nums, left, pivot - 1)
    quick_sort(nums, pivot + 1, right)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick Sort Visualizer')
    parser.add_argument('--low', type=int, default=1, help='Lower bound for random numbers')
    parser.add_argument('--up', type=int, default=9000, help='Upper bound for random numbers')
    parser.add_argument('--leng', type=int, default=40, help='Length of the array')

    args = parser.parse_args()
    arr = random.sample(range(args.low, args.up), args.leng)
    cprint("Original array:", "light_magenta")
    cprint(arr, "light_magenta")
    with open('input_result.txt', 'a') as f:
        f.write(f"Original array: {arr}\n")

    quick_sort(arr, 0, len(arr) - 1)

    cprint("Sorted array:", "cyan")
    cprint(arr, "cyan")  
    with open('input_result.txt', 'a') as f:
        f.write(f"Sorted array: {arr}\n\n")