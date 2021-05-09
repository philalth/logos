import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

from logos.algorithms.Spectacl import Spectacl
from logos.datasets.datasets import load_two_circles
from plotting import plotter
from utils.data_utils import undirected_neighbors_graph
from utils.heuristics import spectacl_eps_heuristic
from utils.matrix_logarithm import matrix_logarithm


def main(show=True):
    # ============
    # Create data
    # ============

    datapoints, true_labels = load_two_circles()
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
    weighted_snn_matrix = undirected_neighbors_graph(datapoints, 14, mode='distance')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        log_weighted_snn_matrix = numpy.array(matrix_logarithm(weighted_snn_matrix), dtype=float)

    weighted_snn_matrix = 1 / (1 + weighted_snn_matrix)

    # ============
    # Compute logos
    # ============

    # Hierarchical logos with weighted symmetric kNN matrix
    weighted_snn_agglom = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='single')
    weighted_snn_agglom.fit(weighted_snn_matrix)
    weighted_snn_agglom_clustering = weighted_snn_agglom.labels_

    # Hierarchical logos with weighted symmetric kNN matrix
    log_agglom = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='single')
    log_agglom.fit(1 / (1 + log_weighted_snn_matrix))
    log_agglom_clustering = log_agglom.labels_

    # SpectACl
    spectacl = Spectacl(n_clusters=clusters, epsilon=spectacl_eps_heuristic(datapoints))
    spectacl.fit(datapoints)
    spectacl_clustering = spectacl.labels_

    # Inverse Heat Clustering
    vals, vecs = numpy.linalg.eig(-1 * log_dist_matrix_rounded)
    vecs = vecs[:, numpy.argsort(vals)]
    clusterer = KMeans(n_clusters=clusters)
    clusterer.fit(vecs[:, 1:clusters])
    inverse_heat_clustering = clusterer.labels_

    clustering_results = (
        ('HC w sym kNN', weighted_snn_agglom_clustering),
        ('HC log w sym kNN', log_agglom_clustering),
        ('SpectACl', spectacl_clustering),
        ('IHC', inverse_heat_clustering)
    )

    if show:
        plotter.plot_data_and_matrices(data, weighted_snn_matrix, log_weighted_snn_matrix)
        plotter.plot_clustering(clustering_results, datapoints)
        plotter.compute_and_plot_evaluation(true_labels, clustering_results)
        pyplot.show()


if __name__ == '__main__':
    main(show=True)
