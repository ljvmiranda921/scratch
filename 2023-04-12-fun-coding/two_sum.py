from typing import List
from wasabi import msg


def variation_1():
    problem = """
    Given an array of integers, return indices of the two numbers such that they
    add up to a specific target.

    You may assume taht each input would have exactly one solution, and you may
    not use the same element twice.

    Input: nums = [2, 7, 11, 5], target = 9
    Output: [0, 1]
    """

    print(problem)

    def solution(nums: List[int], target: int) -> List[int]:
        # Use a hashtable instead of doing pairs itertools.product
        # What would be the target needed to get the value we want?
        cache = {}
        for idx in range(len(nums)):
            # Is the value in the lookup table?
            if nums[idx] in cache:
                return [cache[nums[idx]], idx]
            else:
                cache[target - nums[idx]] = idx
        return None

    print(solution(nums=[2, 7, 11, 15], target=9))


if __name__ == "__main__":
    variation_1()
    msg.divider()
