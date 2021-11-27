import numpy as np


def coef_calculate(alpha, alphas, coefs):
    def coef_calculate_circle(alpha_):
        i = np.sum(alphas > alpha_)
        if i == 0:
            return coefs[0]
        else:
            return (alpha_ - alphas[i]) * (coefs[i - 1] - coefs[i]) / (
                alphas[i - 1] - alphas[i]) + coefs[i]
    
    if type(alpha) == list:                 
        return np.array(
            [coef_calculate_circle(alpha_) for alpha_ in alpha])
    
    else:
        return coef_calculate_circle(alpha)