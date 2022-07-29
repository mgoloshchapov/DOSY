import matplotlib.pyplot as plt
import numpy as np

from src.errors import bootstrap_resudial
from src.mixture_fit import fits
from src.optimal_number import optimal_params


def plot(x, y, title=None, fontsize=15):
    plt.scatter(x, y, color="red", s=10, label="data")
    plt.ylabel('$I/I_0$', fontsize=fontsize)
    plt.xlabel('Z * 1e-6', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)


def param_print(array):
    print("(w, D)")
    print("-----------------")
    for a in array:
        for i in range(0, len(a), 2):
            print((a[i], a[i + 1]))
        print()


def metrics_plot(aics, aic_probs, bics, bic_probs):
    plt.subplot(121)
    plt.plot(range(1, len(aics) + 1), aic_probs, '.')
    plt.hlines(0.32, 1, len(aics) + 1, 'r', alpha=0.5)
    plt.hlines(0.05, 1, len(aics) + 1, 'r', alpha=0.5)
    plt.ylabel('exp($\Delta$AIC/2)')
    plt.xlabel('number of exponents')
    plt.title("AIC")

    plt.subplot(122)
    plt.plot(range(1, len(bics) + 1), bic_probs, '.')
    plt.hlines(0.32, 1, len(bics) + 1, 'r', alpha=0.5)
    plt.hlines(0.05, 1, len(bics) + 1, 'r', alpha=0.5)
    plt.ylabel('exp($\Delta$BIC/2)')
    plt.xlabel('number of exponents')
    plt.title("BIC")
    plt.show()


def number_analysis(x, y, n_min=1, n_max=3, method="BFGS", reg=0.005):
    params = fits(x, y, n_min, n_max, method, reg)
    aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_number = optimal_params(x, y, params)
    metrics_plot(aics, aic_probs, bics, bic_probs)

    print(f"{method}")
    print("---------------------------")
    param_print(params)
    print("---------------------------")
    print(f"AIC: {m_aic + 1}")
    print(f"BIC: {m_bic + 1}")
    print(f"conservative: {cons_number + 1}")

    return params, m_aic + 1, m_bic + 1, cons_number + 1


def error_analysis(n, x, y, method='BFGS', reg=0.0,
                   bs_iters=1000, bs_method='residuals', seed=42):
    init_theta, thetas, res = bootstrap_resudial(n, x, y, bs_iters, bs_method,
                                                 method, seed, reg)
    print(f'estimate: {init_theta}')
    print(f'mean of samples: {np.mean(thetas, axis=0)}')
    print(f'std of samples: {np.std(thetas, axis=0)}')
    return init_theta, thetas, res
