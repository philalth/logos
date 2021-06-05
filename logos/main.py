import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics.pairwise import euclidean_distances

from algorithms.Spectacl import Spectacl
from plotting import figures
from utils import heuristics
from utils.matrix_logarithm import matrix_logarithm


def main(show=True):
    # ============
    # Create data and set parameters
    # ============

    data, labels = fetch_olivetti_faces(return_X_y=True)
    n_features = 4096
    n_clusters = 40
    filter_value = 0.12
    eps = heuristics.dbscan_eps_heuristic(data, 20)

    # ============
    # Compute affinity matrices and their logarithms
    # ============

    # Euclidean distances
    euclidean_distance_matrix = euclidean_distances(data, data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        euclidean_distance_log_matrix = numpy.array(matrix_logarithm(euclidean_distance_matrix),
                                                    dtype=float)

    euclidean_distance_log_filter_matrix = numpy.asarray(
        [[x if x > filter_value else 0 for x in column] for column in
         euclidean_distance_log_matrix])

    # ============
    # Compute clustering
    # ============

    # k-Means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    kmeans_clustering = kmeans.labels_

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=2 * n_features)
    dbscan.fit(data)
    dbscan_clustering = dbscan.labels_

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        # Spectral clustering
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        spectral.fit(data)
        spectral_clustering = spectral.labels_

        # Logos
        logos = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        logos.fit(euclidean_distance_log_filter_matrix)
        logos_clustering = logos.labels_

    # SpectACl
    spectacl = Spectacl(n_clusters=n_clusters, epsilon=heuristics.spectacl_eps_heuristic(data))
    spectacl.fit(data)
    spectacl_clustering = spectacl.labels_

    algorithms = (
        ('Logos', logos_clustering),
        ('k-Means', kmeans_clustering),
        ('DBSCAN', dbscan_clustering),
        ('Spectral\nClustering', spectral_clustering),
        ('SpectACl', spectacl_clustering),
    )

    if show:
        figures.compute_and_plot_evaluation(labels, algorithms)
        pyplot.show()


if __name__ == '__main__':
    main(show=True)
