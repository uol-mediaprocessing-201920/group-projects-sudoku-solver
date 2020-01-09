import time
import multiprocessing


# def mp_solve_process(queue):
#     grid = queue.get()
#     result = solve(grid)
#     queue.put(result)
#
#
# def mp_solve(grid, timeout):
#     queue = multiprocessing.Queue()
#     process = multiprocessing.Process(target=mp_solve_process, args=(queue,))
#     process.start()
#     queue.put(grid)
#     process.join(timeout=timeout)
#     process.kill()
#     result = queue.get()
#     return result
#
#
# stolen from https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/

def solve(grid, start=None, timeout=None):
    if timeout is not None:
        if start is None:
            start = time.time()
        elif time.time() - start > timeout:
            raise TimeoutError()

    # try find empty cell
    find = find_empty(grid)
    if not find:
        # no empty cell found
        # sudoku is solved
        return True
    else:
        # empty cell found
        # continue solving
        row, col = find

    box_row = int(row / 3)
    box_col = int(col / 3)
    numbers_in_box = grid[box_row * 3:(box_row + 1) * 3, box_col * 3:(box_col + 1) * 3]
    numbers_in_row = grid[row, :]
    numbers_in_col = grid[:, col]
    possible_numbers = [i for i in range(1, 10) if i not in numbers_in_row and i not in numbers_in_col and i not in numbers_in_box]

    # try each number
    for i in possible_numbers:
        # check if sudoku is still valid
        if valid(grid, i, (row, col)):
            # sudoku is still valid, keep solving
            grid[row][col] = i
            if solve(grid, start, timeout):
                # sudoku has been successfully solved somehow
                return True
            else:
                # did not find solution
                # reset to empty
                grid[row][col] = 0
    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None
