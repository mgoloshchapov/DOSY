import numpy as np


def exp(x, params):
    w, D = params
    return w * np.exp(-D * x)


def exp_mixture(x, exp_params):
    return sum(exp(x, params) for params in exp_params)


def loss_function(params, x, y, n, func=exp_mixture):
    params = params.reshape(n, 2)
    return np.sqrt(np.sum(y - func(x, params)) ** 2)
