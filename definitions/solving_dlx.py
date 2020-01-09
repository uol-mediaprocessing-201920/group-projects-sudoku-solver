# https://github.com/jjallan/sudoku/blob/master/sudoku.py

import dlx
import numpy as np
from itertools import permutations, takewhile
from random import choice, shuffle

'''
A 9 by 9 sudoku solver.
'''
_N = 3
_NSQ = _N ** 2
_NQU = _N ** 4
_VALID_VALUE_INTS = list(range(1, _NSQ + 1))
_VALID_VALUE_STRS = [str(v) for v in _VALID_VALUE_INTS]
_EMPTY_CELL_CHAR = '·'

# The following are mutually related by their ordering, and define ordering throughout the rest of the code. Here be dragons.
#
_CANDIDATES = [(r, c, v) for r in range(_NSQ) for c in range(_NSQ) for v in range(1, _NSQ + 1)]
_CONSTRAINT_INDEXES_FROM_CANDIDATE = lambda r, c, v: [_NSQ * r + c, _NQU + _NSQ * r + v - 1,
                                                      _NQU * 2 + _NSQ * c + v - 1,
                                                      _NQU * 3 + _NSQ * (_N * (r // _N) + c // _N) + v - 1]
_CONSTRAINT_FORMATTERS = ["R{0}C{1}", "R{0}#{1}", "C{0}#{1}", "B{0}#{1}"]
_CONSTRAINT_NAMES = [(s.format(a, b + (e and 1)), dlx.DLX.PRIMARY) for e, s in enumerate(_CONSTRAINT_FORMATTERS) for a
                     in range(_NSQ) for b in range(_NSQ)]
_EMPTY_GRID_CONSTRAINT_INDEXES = [_CONSTRAINT_INDEXES_FROM_CANDIDATE(r, c, v) for (r, c, v) in _CANDIDATES]


#
# The above are mutually related by their ordering, and define ordering throughout the rest of the code. Here be dragons.


class Solver:
    def __init__(self, representation=''):
        if not representation or len(representation) != _NQU:
            self._complete = False
            self._NClues = 0
            self._repr = [
                             None] * _NQU  # blank grid, no clues - maybe to extend to a generator by overriding the DLX column selection to be stochastic.
        else:
            nClues = 0
            repr = []
            for value in representation:
                if not value:
                    repr.append(None)
                elif isinstance(value, int) and 1 <= value <= _NSQ:
                    nClues += 1
                    repr.append(value)
                elif value in _VALID_VALUE_STRS:
                    nClues += 1
                    repr.append(int(value))
                else:
                    repr.append(None)
            self._complete = nClues == _NQU
            self._NClues = nClues
            self._repr = repr

    def genSolutions(self, genSudoku=True, genNone=False, dlxColumnSelctor=None):
        '''
        if genSudoku=False, generates each solution as a list of cell values (left-right, top-bottom)
        '''
        if self._complete:
            yield self
        else:
            self._initDlx()
            dlxColumnSelctor = dlxColumnSelctor or dlx.DLX.smallestColumnSelector
            if genSudoku:
                for solution in self._dlx.solve(dlxColumnSelctor):
                    yield Solver([v for (r, c, v) in sorted([self._dlx.N[i] for i in solution])])
            elif genNone:
                for solution in self._dlx.solve(dlxColumnSelctor):
                    yield
            else:
                for solution in self._dlx.solve(dlxColumnSelctor):
                    yield [v for (r, c, v) in sorted([self._dlx.N[i] for i in solution])]

    def uniqueness(self, returnSolutionIfProper=False):
        '''
        Returns: 0 if unsolvable;
                 1 (or the unique solution if returnSolutionIfProper=True) if uniquely solvable; or
                 2 if multiple possible solutions exist
        - a 'proper' sudoku is uniquely solvable.
        '''
        slns = list(takewhile(lambda t: t[0] < 2, ((i, sln) for i, sln in enumerate(
            self.genSolutions(genSudoku=returnSolutionIfProper, genNone=not returnSolutionIfProper)))))
        uniqueness = len(slns)
        if returnSolutionIfProper and uniqueness == 1:
            return slns[0][1]
        else:
            return uniqueness

    def representation(self, asString=True, noneCharacter='.'):
        if asString:
            return ''.join([v and str(_VALID_VALUE_STRS[v - 1]) or noneCharacter for v in self._repr])
        return self._repr[:]

    def __repr__(self):
        return display(self._repr)

    def _initDlx(self):
        self._dlx = dlx.DLX(_CONSTRAINT_NAMES)
        rowIndexes = self._dlx.appendRows(_EMPTY_GRID_CONSTRAINT_INDEXES, _CANDIDATES)
        for r in range(_NSQ):
            for c in range(_NSQ):
                v = self._repr[_NSQ * r + c]
                if v is not None:
                    self._dlx.useRow(rowIndexes[_NQU * r + _NSQ * c + v - 1])


def solve(grid, n_rows=9, n_cols=9):
    try:
        solver_input = grid.flatten().tolist()
        solution = next(Solver(solver_input).genSolutions())
        solution = solution.representation(asString=False)
        solution = np.array(solution).reshape(n_rows, n_cols)
        return solution
    except StopIteration:
        return False
