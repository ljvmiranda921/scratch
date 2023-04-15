from collections import Counter

from wasabi import msg


def variation_1():
    problem = """
    Given two strings s and t, write a function to determine if t is an anagram
    of s.
    Input: s = "anagram", t = "nagaram"
    Output: true

    Input: s = "rat" t = "car"
    Output: false
    """

    print(problem)

    def solution(s: str, t: str) -> bool:
        # Slower sol'n sorted(s) == sorted(t)
        return Counter(s) == Counter(t)

    print(solution(s="anagram", t="nagaram"))
    print(solution(s="rat", t="car"))


if __name__ == "__main__":
    variation_1()
    msg.divider()
