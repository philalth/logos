import warnings

import numpy
from matplotlib import pyplot
from scipy.linalg import expm, logm
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from utils.matrix_logarithm import matrix_logarithm


def _relative_error_matrix(true_matrix, predicted_matrix):
    if true_matrix.shape != estimated_matrix.shape:
        raise ValueError
    error_matrix = numpy.zeros(true_matrix.shape)
    for i in range(0, len(true_matrix)):
        for j in range(0, len(true_matrix)):
            if i != j:
                error_matrix[i][j] = abs(predicted_matrix[i][j] - true_matrix[i][j]) / true_matrix[i][j]
    return error_matrix


data, labels = make_blobs(n_samples=1000, shuffle=False)

matrix = euclidean_distances(data, data)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', numpy.ComplexWarning)
    log_dist_matrix = numpy.array(matrix_logarithm(matrix), dtype=float)

estimated_matrix = expm(log_dist_matrix)

pyplot.imshow(_relative_error_matrix(matrix, estimated_matrix))
pyplot.colorbar()
pyplot.title('Relative error matrix')
pyplot.show()


mse_values = []  # Mean squared error
sre_values = []  # Summed relative error (1-norm)
bre_values = []  # Biggest relative error

for x in range(5, 100, 1):
    data, labels = make_blobs(n_samples=1000, n_features=2, shuffle=False, center_box=(-x, x))

    matrix = euclidean_distances(data, data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        log_dist_matrix = numpy.array(matrix_logarithm(matrix), dtype=float)

    estimated_matrix = expm(log_dist_matrix)

    mse = mean_squared_error(matrix, estimated_matrix)
    sre = logm(matrix, disp=False)[1]
    bre = numpy.max(estimated_matrix - matrix) / numpy.max(matrix)

    print('mse:', mse)
    print('sre:', sre)
    print('bre:', bre)

    mse_values.append(mse)
    sre_values.append(sre)
    bre_values.append(bre)

print('Mean squared error:', mse_values)
print('1-norm of the estimated error (python):', sre_values)
print('Biggest relative error:', bre_values)
