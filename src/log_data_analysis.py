import numpy as np
from scipy.optimize import Bounds


def log_estimate(x, y, w_min=0.01):
    """
    This function estimates parameters of exponent with the lowest an absolute degree and the greatest one
    :param x:
    :param y:
    :param w_min:
    :return:
    """
    cut = round(x.size / 4)

    D_1, w_1 = np.polyfit(x[-cut:], np.log(y)[-cut:], deg=1)
    D_1, w_1 = -D_1, np.exp(w_1)

    D_n, b = np.polyfit(x[:cut], y[:cut], deg=1)
    D_n = -D_n

    return D_1, min(D_n / w_min, 1e-5), w_1


def bounds(D1, w1, n, D_max, w_min):
    D_min = D1*0.9
    w_max = 1 - w1
    # initial guess
    Ds = np.linspace(D1, D_max * 0.9, n)
    ws = np.zeros(n)
    ws[0] = w1
    ws[1:] = np.linspace(-w_max * 0.9, -w_min * 1.1, n - 1)
    ws[1:] = -ws[1:]
    x0 = np.zeros((n, 2))
    x0[:, 0] = ws
    x0[:, 1] = Ds
    x0 = x0.flatten()
    # lower bound
    xl = np.zeros((n, 2))
    xl[:, 0] = w_min
    xl[:, 1] = D_min
    xl = xl.flatten()
    # upper bound
    xw = np.zeros((n, 2))
    xw[:, 0] = 1
    xw[:, 1] = D_max
    xw = xw.flatten()
    bnds = Bounds(tuple(xl), tuple(xw))
    return x0, bnds

