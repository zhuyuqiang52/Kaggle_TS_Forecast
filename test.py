import sys
import numpy as np
#sys.setrecursionlimit(1000000)
#mat_list = np.random.randint(5,size=90).reshape(9,10).tolist()
#nums = list(range(len(mat_list)))
#sum_list = [sum(i) for i in mat_list]
c = ord('a')
def quick_sort(beg, end, nums, sum_list):
    if beg + 1 > end:
        return
    l, r = beg, end
    key = nums[beg]
    while beg < end:
        while sum_list[nums[end]] > sum_list[key] or ((sum_list[nums[end]] == sum_list[key]) & (nums[end] > key)):
            end -= 1
        nums[beg], nums[end] = nums[end], nums[beg]
        while sum_list[nums[beg]] < sum_list[key] or ((sum_list[nums[beg]] == sum_list[key]) & (nums[beg] < key)):
            beg += 1
        nums[beg], nums[end] = nums[end], nums[beg]
    # left sort
    quick_sort(l, beg, nums, sum_list)
    # right sort
    quick_sort(end, r, nums, sum_list)
#quick_sort(0,1,nums,sum_list)
#print(nums)

a = np.random.randint(9,size=10)
b = np.argsort(a)
help(np.argsort)
def quick_sort2(beg, end, nums):
    if beg + 1 > end:
        return
    l, r = beg, end
    key = nums[beg]
    while beg < end:
        while nums[end] > key or (nums[end] == key and end>l):
            end -= 1
        nums[end], nums[beg] = nums[beg], nums[end]
        while beg < end and nums[beg] < key or (nums[beg] == key and beg<r):
            beg += 1
        nums[end], nums[beg] = nums[beg], nums[end]
    # left sort
    quick_sort2(l, end, nums)
    quick_sort2(end + 1, r, nums)



def maximumWealth(accounts) -> int:
    sum_acc = [sum(list_i) for list_i in accounts]
    end = len(sum_acc) - 1
    quick_sort2(0, end, sum_acc)
    return sum_acc[-1]


class hash_map:
    def __init__(self):
        self.hash_len_int = 2 ^ 10
        self.hash_list = [0] * self.hash_len_int

    def put(self, idx, i):
        loc_int = i % self.hash_len_int
        if self.hash_list[loc_int] == 0:
            self.hash_list[loc_int] = [idx]
        else:
            self.hash_list[loc_int].append(idx)

    def has(self, i,nums):
        loc_int = i % self.hash_len_int
        if self.hash_list[loc_int]!=0:
            for idx in self.hash_list[loc_int]:
                if nums[idx]==i:
                    return idx
        return -1


def twoSum(nums, target: int):
    num_hashmap = hash_map()
    for i in range(len(nums)):
        loc = num_hashmap.has(target - nums[i],nums)
        if loc != -1:
            return [loc, i]
        num_hashmap.put(i, nums[i])
nums = [1,6142,8192,10239]
target = 18431
twoSum(nums=nums,target=target)