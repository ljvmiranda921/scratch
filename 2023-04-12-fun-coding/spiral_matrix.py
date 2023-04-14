from typing import List
from wasabi import msg


def variation_1():
    problem = """
    Given an m x n matrix, return all elements of the matrix in spiral order
    Input: matrix = [[1,2,3], [4,5,6], [7,8,9]]
    Output: [1,2,3,6,9,8,7,4,5]
    """
    print(problem)

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def solution(matrix: List[List[int]]) -> List[int]:
        # Edge case where they give you an empty matrix
        if not matrix:
            return []

        # Then we're tracking four variables to traverse the matrix
        start_row, end_row = 0, len(matrix)
        start_col, end_col = 0, len(matrix[0])

        output = []

        # We will run through a while loop,
        # while the starting row is less than the ending row
        while start_row < end_row or start_col < end_col:
            # Each time we change directions, we're going to change
            # our start/ending row and start/ending col so that we don't append
            # the values we have already traversed.

            # However, there's an edge-case: if we have a non-square matrix
            # (4x3, 3x4), this wouldn't work because the conditions in the while loop
            # won't work. The solution is to add the if-statements

            # fmt: off
            # Going right
            if start_row < end_row:
                output.extend([matrix[start_row][i] for i in range(start_col, end_col)])
                start_row += 1
            # Going down
            if start_col < end_col:
                output.extend([matrix[i][end_col - 1] for i in range(start_row, end_row)])
                end_col -= 1
            # Going left
            if start_row < end_row:
                output.extend([matrix[end_row - 1][i] for i in range(end_col - 1, start_col - 1, -1)])
                end_row -= 1
            # Going up
            if start_col < end_col:
                output.extend([matrix[i][start_col] for i in range(end_row - 1, start_row - 1, -1)])
                start_col += 1
            # fmt: on

        return output

    print(solution(matrix))


def variation_2():
    problem = """
    Given a positive integer n, generate an n x n matrix filled with elements from 1 to n**2
    in spiral order
    Input: n = 3
    Output: [[1,2,3],[8,9,4],[7,6,5]]
    """
    print(problem)

    n = 3

    def solution(n: int) -> List[List[int]]:
        # keep track of four edges, start/end row/col
        # as long as the start col < end_col, start_row , end_row
        # we're going to do four for-loops. will fill in the correct order: R, D, L, U
        # each time we make 1-fill, we increment/decrement stuff, to make sure we don't overwrite
        output = [[0] * n for _ in range(n)]
        val = 0
        start_row, end_row = 0, n
        start_col, end_col = 0, n

        while start_col < end_col or start_row < end_row:
            # Going right
            for i in range(start_col, end_col):
                val += 1
                output[start_row][i] = val
            start_row += 1
            # Going down
            for i in range(start_row, end_row):
                val += 1
                output[i][end_col - 1] = val
            end_col -= 1
            # Going left
            for i in range(end_col - 1, start_col - 1, -1):
                val += 1
                output[end_row - 1][i] = val
            end_row -= 1
            # Going up
            for i in range(end_row - 1, start_row - 1, -1):
                val += 1
                output[i][start_col] = val
            start_col += 1

        return output

    print(solution(n))


if __name__ == "__main__":
    variation_1()
    msg.divider()
    variation_2()
