from collections import Counter

from wasabi import msg


def anagram():
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


def longest_substring():
    problem = """
    Given a string, find the length of the longest substring without repeating characters
    Input: "abcabcbb
    Output: 3

    Input: "bbbbb
    Output: 1

    Input: "pwwkew"
    Output: 3
    """
    print(problem)

    def solution(s: str) -> int:
        start = 0
        maxlen = 0
        lookup = {}

        for idx, char in enumerate(s):
            # Have we seen this character before?
            if char in lookup and start <= lookup[char]:
                start = lookup[char] + 1
            else:
                maxlen = max(maxlen, idx - start + 1)
            lookup[char] = idx

    print([solution(qxn) for qxn in ("abcabcbb", "bbbbb", "pwwkew")])


if __name__ == "__main__":
    anagram()
    msg.divider()