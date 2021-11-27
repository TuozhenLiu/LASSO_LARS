import numpy as np
from scipy import linalg
from src.solve_sym import solve_sym

def LARS_pure(xtx, xty, x_mean, y_meaLARSn, y_std, n, p, standarize, fit_intercept):
    # 标准化
    if standarize:
        xtx = xtx - np.outer(x_mean, x_mean) * n  # 中心化
        xty = xty - x_mean * y_mean * n  # 中心化
        x_std = np.sqrt(np.diag(xtx) / (n - 1))  # x标准差 p维
        x_std_mat = 1 / np.repeat(x_std.reshape((1, p)), p, axis=0)  # x标准差的倒数矩阵 pxp维
        xtx = x_std_mat.T * xtx * x_std_mat  # 标准化后的xTx（相当于用x除以std后再xTx）
        xty = xty / x_std / y_std # 标准化后的xTy
    
    # 初始化
    A = np.array([False] * p)             # 活跃集
    coef = np.zeros(p, dtype=float)       # 系数
    ck = xty                              # 活跃变量XAk与残差的内积
    ck_abs = np.absolute(ck)
    j = np.argmax(ck_abs)
    lamb = ck_abs[j]                       # lambda0
    lambs = [lamb]
    coefs = [np.copy(coef)]
    A[j] = True                           # 进入活跃集
    
    # 迭代
    while (True):
        n_Ak = np.sum(A)                  # 活跃集变量个数
        d = solve_sym(xtx[A][:, A], np.sign(ck)[A] ) 
        gam = np.ones(p, dtype=float)     # 步长gamma
        
        # 考虑变量进入
        if (n_Ak < p):
            ak = np.dot(xtx[~A][:, A], d)  
            gam[~A] = np.where(ak * lamb <= ck[~A], (lamb - ck[~A]) / (1 - ak),
                            (lamb + ck[~A]) / (1 + ak))

        # 考虑变量退出
        if (n_Ak > 0):
            w = -coef[A] / d
            gam[A] = np.where(((w > 0) & (w < lamb)), w, lamb)  
        
        # 更新
        j = np.argmin(gam)
        gam_min = gam[j]
        coef[A] = coef[A] + gam_min * d
        lamb = lamb - gam_min
        lambs.append(lamb)
        coefs.append(np.copy(coef))
        if (lamb == 0):
            break
        A[j] = ~A[j]
        ck = xty - np.dot(xtx, coef)
    
    # 还原系数
    if standarize:
        coefs = np.array(coefs) / x_std * y_std
        if fit_intercept:
            coefs = np.c_[(y_mean - np.dot(coefs, x_mean)).reshape((-1, 1)), coefs]
    
    alphas = np.array(lambs) / n
    return alphas, coefs

