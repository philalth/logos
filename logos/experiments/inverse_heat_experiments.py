import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from logos.datasets import datasets
from plotting import plotter
from utils.data_utils import undirected_neighbors_graph
from utils.matrix_logarithm import matrix_logarithm


def main(show=True):
    # ============
    # Create data
    # ============

    datapoints, true_labels = datasets.load_two_circles()
    data = datapoints, true_labels

    clusters = 2

    # ============
    # Compute affinity matrices and their logarithms
    # ============

    # Euclidean distances
    distance_matrix = euclidean_distances(datapoints, datapoints)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        log_distance_matrix = numpy.array(matrix_logarithm(distance_matrix), dtype=float)

    log_dist_matrix_rounded = numpy.asarray([[x if x > 0.05 else 0 for x in column] for column in log_distance_matrix])

    # Symmetric k-nearest-neighbours
    unweighted_snn_matrix = undirected_neighbors_graph(datapoints, 14, mode='connectivity')
    weighted_snn_matrix = undirected_neighbors_graph(datapoints, 14, mode='distance')
    # weighted_snn_matrix = 1 / (1 + weighted_snn_matrix)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        log_unweighted_snn_matrix = numpy.array(matrix_logarithm(unweighted_snn_matrix), dtype=float)
        log_weighted_snn_matrix = numpy.array(matrix_logarithm(weighted_snn_matrix), dtype=float)

    # ============
    # Compute logos
    # ============

    clustering_results = (
        ('IHC euclid log', inverse_heat_clustering(log_distance_matrix, clusters)),
        # ('IHC sym uw kNN log', inverse_heat_clustering(log_unweighted_snn_matrix, clusters)),
        ('IHC sym w kNN log', inverse_heat_clustering(log_weighted_snn_matrix, clusters)),
    )

    if show:
        plotter.plot_data_and_matrices(data, weighted_snn_matrix, log_weighted_snn_matrix)
        plotter.plot_clustering(clustering_results, datapoints)
        plotter.compute_and_plot_evaluation(true_labels, clustering_results)
        pyplot.show()


def inverse_heat_clustering(matrix, n_clusters):
    eigenvalues, eigenvectors = numpy.linalg.eig(-1 * matrix)
    eigenvectors = eigenvectors[:, numpy.argsort(eigenvalues)]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(eigenvectors[:, 1:n_clusters])
    return kmeans.labels_


if __name__ == '__main__':
    main(show=True)
