from typing import List


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


if __name__ == "__main__":
    variation_1()
