import matplotlib.pyplot as plt

from src.data_loading import load_data
from src.mixture_fit import fits, sum_exp
from src.optimal_number import AIC_analysis, BIC_analysis


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


def metrics_plot(x, y, params, s, method):
    if method == "curve_fit":
        plt.figure(figsize=(10, 5))
        m_aic, aics, aic_probs = AIC_analysis(x, y, params, s)

        plt.subplot(121)
        plt.plot(range(1, len(aics) + 1), aic_probs, '.')
        plt.hlines(0.32, 1, len(aics) + 1, 'r', alpha=0.5)
        plt.hlines(0.05, 1, len(aics) + 1, 'r', alpha=0.5)
        plt.ylabel('exp($\Delta$AIC/2)')
        plt.xlabel('number of exponents')
        plt.title("Curve fit AIC")

        m_bic, bics, bic_probs = BIC_analysis(x, y, params, s)

        plt.subplot(122)
        plt.plot(range(1, len(bics) + 1), bic_probs, '.')
        plt.hlines(0.32, 1, len(bics) + 1, 'r', alpha=0.5)
        plt.hlines(0.05, 1, len(bics) + 1, 'r', alpha=0.5)
        plt.ylabel('exp($\Delta$BIC/2)')
        plt.xlabel('number of exponents')
        plt.title("Curve fit BIC")

        plt.show()

    elif method == "dual_annealing":
        plt.figure(figsize=(10, 5))
        m_aic, aics, aic_probs = AIC_analysis(x, y, params, s)

        plt.subplot(121)
        plt.plot(range(1, len(aics) + 1), aic_probs, '.')
        plt.hlines(0.32, 1, len(aics) + 1, 'r', alpha=0.5)
        plt.hlines(0.05, 1, len(aics) + 1, 'r', alpha=0.5)
        plt.ylabel('exp($\Delta$AIC/2)')
        plt.xlabel('number of exponents')
        plt.title("Dual annealing AIC")

        m_bic, bics, bic_probs = BIC_analysis(x, y, params, s)

        plt.subplot(122)
        plt.plot(range(1, len(bics) + 1), bic_probs, '.')
        plt.hlines(0.32, 1, len(bics) + 1, 'r', alpha=0.5)
        plt.hlines(0.05, 1, len(bics) + 1, 'r', alpha=0.5)
        plt.ylabel('exp($\Delta$BIC/2)')
        plt.xlabel('number of exponents')
        plt.title("Dual annealing BIC")

        plt.tight_layout()
        plt.show()

    else:
        raise "InvalidMethod. [curve_fit, dual_anneling] are available"

    return m_aic, m_bic


def graphics_plot(x, y, params, m_aic, m_bic, method):
    if method == "curve_fit":
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(x, sum_exp(params[m_aic], x))
        plot(x, y, "{} exponents, curve fit, AIC".format(m_aic + 1))

        plt.subplot(122)
        plt.plot(x, sum_exp(params[m_bic], x))
        plot(x, y, "{} exponents, curve fit, BIC".format(m_bic + 1))
        plt.tight_layout()
        plt.show()

    elif method == "dual_annealing":
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.plot(x, sum_exp(params[m_aic], x))
        plot(x, y, "{} exponents, dual annealing, AIC".format(m_aic + 1))

        plt.subplot(122)
        plt.plot(x, sum_exp(params[m_bic], x))
        plot(x, y, "{} exponents, dual annealing, BIC".format(m_bic + 1))
        plt.tight_layout()
        plt.show()

    else:
        raise "InvalidMethod. [curve_fit, dual_anneling] are available"


def dosy_analysis(path, n_min=1, n_max=3, method="curve_fit"):
    x, y = load_data(path, scale=1e6)

    params_cf, params_da = None, None
    if method == "curve_fit" or method == "both":
        params_cf, s_cf = fits(x, y, n_min=n_min, n_max=n_max, method="curve_fit")

        m_cf_aic, m_cf_bic = metrics_plot(x, y, params_cf, s_cf, method="curve_fit")

        graphics_plot(x, y, params_cf, m_cf_aic, m_cf_bic, method="curve_fit")

        print()

        print("Curve fit")
        print("---------------------------")
        param_print(params_cf)

    if method == "dual_annealing" or method == "both":
        params_da, s_da = fits(x, y, n_min=n_min, n_max=n_max, method="dual_annealing")

        m_da_aic, m_da_bic = metrics_plot(x, y, params_da, s_da, method="dual_annealing")

        graphics_plot(x, y, params_da, m_da_aic, m_da_bic, method="dual_annealing")

        print()

        print("Dual annealing")
        print("---------------------------")
        param_print(params_da)

    if method not in ["curve_fit", "dual_annealing", "both"]:
        raise "InvalidMethod. [curve_fit, dual_anneling, both] are available"

    return [x, y, params_cf, params_da]
