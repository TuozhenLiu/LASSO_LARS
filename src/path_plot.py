import numpy as np
import matplotlib.pyplot as plt


def path_plot(coefs):
    xx = np.sum(np.abs(coefs), axis=1)
    xx /= xx[-1]
    plt.plot(xx, coefs)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()