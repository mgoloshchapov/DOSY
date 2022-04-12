import numpy as np


def load_data(link_to_data: str,
              skiprows: int = 2,
              usecols: tuple = (1, 2)) -> tuple[np.ndarray, np.ndarray]:
    """
    This function load data from single txt file
    :param link_to_data: link to txt file with data
    :param skiprows: number of rows in the file to skip
    :param usecols: indexes of columns to use
    :return:
    """
    x, y = np.loadtxt(link_to_data, skiprows=skiprows, usecols=usecols).T
    # we consider I/I0
    x = x / y[0]
    y = y / y[0]
    return x, y
