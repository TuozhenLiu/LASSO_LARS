from scipy import linalg


def solve_sym(xtx, xty):
    from scipy import linalg
    L = linalg.cholesky(xtx)  # scipy包上三角矩阵
    return linalg.lapack.dpotrs(L, xty)[0]