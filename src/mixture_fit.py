import numpy as np
from scipy import optimize

from src.log_data_analysis import log_estimate, bounds


def sum_exp(params, x):
    """
    Exponent sum
    :param params: Starting with 1, odd params - weights of exponents, even params - coefficients
    """
    res = np.zeros(len(x))
    for i in range(0, len(params), 2):
        res += params[i] * np.exp(-x * params[i + 1])
    return res


def sum_exp_curv(x, *params):
    return sum_exp(params, x)


def least_sqruares(y, y_pred):
    return np.linalg.norm(y_pred - y)


def chi_square(y, y_pred, sigma):
    dof = len(y)
    return np.sum((y - y_pred) ** 2) / sigma ** 2 / dof


def gen_data(seed, params, n=225, sigma=0.001):
    np.random.seed(seed)
    x = np.geomspace(0.1, 6, n)
    y = np.random.normal(loc=sum_exp(params, x), scale=sigma)
    y[y < 0] = -y[y < 0]
    return x, y


def right_order(params):
    """
    This function makes the order of exponents in the right way:
    little coefficients are first
    """
    sort_params = np.zeros(len(params))
    sort_ind = np.argsort(params[1::2])
    sort_params[1::2] = params[2 * sort_ind + 1]
    sort_params[::2] = params[2 * sort_ind]
    return sort_params


def loss_function(params, x, y, reg=0, func=sum_exp):
    y_pred = func(params, x)
    return least_sqruares(y, y_pred) + reg * np.linalg.norm(params)


def fits(x, y, n_min=1, n_max=5, method='curve_fit', reg=0.01):

    w1, D1, D_max, s = log_estimate(x, y)
    params_est = []

    for n in range(n_min, n_max + 1):

        x0, xl, xw = bounds(D1, w1, D_max, 2 * n)

        if method == 'curve_fit':
            params, pcov = optimize.curve_fit(sum_exp_curv, x, y, p0=x0, bounds=(xl, xw),
                                              maxfev=100000)


        elif method == 'dual_annealing':
            res = optimize.dual_annealing(loss_function, bounds=list(zip(xl, xw)), x0=x0,
                                          args=(x, y, reg),
                                          seed=42, initial_temp=1, maxiter=1000, visit=2, accept=-1,
                                          no_local_search=False,
                                          minimizer_kwargs={'method': 'BFGS'})
            params = res.x


        else:
            raise ValueError('method should be curve_fit or dual_annealing')

        params_est.append(right_order(params))

    return params_est, s
