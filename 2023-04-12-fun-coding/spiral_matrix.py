def variation_1():
    problem = """
    Given an m x n matrix, return all elements of the matrix in spiral order
    Input: matrix = [[1,2,3], [4,5,6], [7,8,9]]
    Output: [1,2,3,6,9,8,7,4,5]
    """
    print(problem)

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def solution(m):
        output = []
        if len(m) == 0:
            return output

        # Prepare the indices you will traverse on
        row_start, col_start = 0, 0
        row_end, col_end = len(m) - 1, len(m[0]) - 1

        while row_start <= row_end and col_start <= col_end:
            for i in range(col_start, col_end + 1):
                output.append(m[row_start][i])
            row_start += 1

            for i in range(row_start, row_end + 1):
                output.append(m[i][col_end])
            col_end -= 1

            if row_start <= row_end:
                for i in range(col_end, col_start - 1, -1):
                    output.append(m[row_end][i])
                row_end -= 1
            if col_start <= col_end:
                for i in range(row_end, row_start - 1, -1):
                    output.append(m[i][col_start])
                col_start += 1
        return output

    print(solution(matrix))


if __name__ == "__main__":
    variation_1()
