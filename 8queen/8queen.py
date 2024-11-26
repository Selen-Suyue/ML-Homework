import numpy as np


class NQueensSolver:
    def __init__(self, n):
        self.n = n  
        self.solutions = []  
        self.board = np.zeros((n, n), dtype=int) 

    def solve(self):
        self._backtrack(0)
        return self.solutions

    def _backtrack(self, row):
        if row == self.n:
            
            self.solutions.append(self.board.copy())
            return
        for col in range(self.n):
            if self._is_valid(row, col):
                self.board[row, col] = 1  
                self._backtrack(row + 1)  
                self.board[row, col] = 0  

    def _is_valid(self, row, col):
        
        for i in range(row):
            if (self.board[i, col] == 1 or  
                    (col - (row - i) >= 0 and self.board[i, col - (row - i)] == 1) or  
                    (col + (row - i) < self.n and self.board[i, col + (row - i)] == 1)):  
                return False
        return True

    def save_solutions(self, filename):
        with open(filename, 'w') as f:
            for index, solution in enumerate(self.solutions):
                f.write(f"Solution {index + 1}:\n")
                for row in solution:
                    f.write(" ".join("Q" if col == 1 else "." for col in row) + "\n")
                f.write("\n")


n = 8  
solver = NQueensSolver(n)
solutions = solver.solve()


solver.save_solutions("eight_queens_solutions.txt")

print(f"Found {len(solutions)} solutions for {n}-Queens problem.")
