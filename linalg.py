from random import randint
import numpy as np



class GF2:
    def __init__(self, A: np.array):
        self.matrix = A % 2
    

    def transpose(self):
        return GF2(np.transpose(self.matrix))

    def leftmul(self, v: np.array):
        m, n = self.matrix.shape
        assert(m == v.size)
        res = []
        for i in range(0, n):
            col = self.matrix[:, i]
            dot_prod = 0
            for j in range(0, m):
                dot_prod += v[j] * col[j]
            res.append(dot_prod % 2)
        return res

    # swaps (zero-indexed) (in-place) c1 with c2
    def colswap(self, c1: int, c2: int):
        self.matrix[:, [c1, c2]] = self.matrix[:, [c2, c1]]

        # adds (zero-indexed) column c2 to column c1
    def add_to_col(self, c1: int, c2: int):
        self.matrix[:, c1] = (self.matrix[:, c1] + self.matrix[:, c2]) % 2
    

    def ker(self) -> np.array:
        m, n = self.matrix.shape

        A = self
        I = GF2(np.identity(n, dtype=int))

        pivot_row, pivot_col = 0, 0
        # go through each row
        while pivot_row < m:
            # try setting nonzero pivot 
            if A.matrix[pivot_row][pivot_col] == 0:
                for col in range(pivot_col + 1, n):
                    if A.matrix[pivot_row][col] != 0:
                        A.colswap(pivot_col, col)
                        I.colswap(pivot_col, col)
                        break

            # zero out everything to the right of pivot
            for col in range(pivot_col + 1, n):
                if A.matrix[pivot_row][col] != 0:
                    A.add_to_col(col, pivot_col)
                    I.add_to_col(col, pivot_col)


            pivot_row += 1
            pivot_col += 1 if pivot_col < n - 2 else 0
    

        basis = []
        for col in range(0,n):
            if not np.any(A.matrix[:, col]):
                basis.append(I.matrix[:, col])

        return np.array(basis)

    def co_ker(self) -> np.array:
        return self.transpose().ker()

    def __repr__(self):
        return str(self.matrix)




def test_gf2(iters: int = 100, debug=False):
    for i in range(iters):
        m = randint(1,20)
        n = randint(1,20)
        A = GF2(np.ceil((np.random.rand(m,n) * 10)).astype(int) % 2)
        ln = A.co_ker()

        if debug:
            print(f"A ({m} x {n}):")
            print(A)
            print("=" * 20)
            print(f"left nullspace of A:")
            print(ln)
            print("=" * 20)

        if ln.size == 0:
            if debug:
                print("No solution")
        else:
            for v in ln:
                should_be_zero = A.leftmul(v)
                if debug:
                    print(f"{v} * A = {should_be_zero}")
                assert(not np.any(should_be_zero))

        print("test passed")
        print("\n" * 3)



