import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics.pairwise import euclidean_distances

from logos.algorithms.Spectacl import Spectacl
from plotting import figures
from utils import heuristics
from utils.data_utils import undirected_neighbors_graph
from utils.matrix_logarithm import matrix_logarithm


def main(show=True):
    # ============
    # Create data and set parameters
    # ============

    data, labels = fetch_olivetti_faces(return_X_y=True)
    n_features = 4096
    n_clusters = 40
    n_neighbours = 10
    filter_value = 0.12
    eps = heuristics.dbscan_eps_heuristic(data, 20)

    # ============
    # Compute affinity matrices and their logarithms
    # ============

    # Euclidean distances
    euclidean_distance_matrix = euclidean_distances(data, data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        euclidean_distance_log_matrix = numpy.array(matrix_logarithm(euclidean_distance_matrix), dtype=float)

    euclidean_distance_log_filter_matrix = numpy.asarray(
        [[x if x > filter_value else 0 for x in column] for column in euclidean_distance_log_matrix])

    # Undirected k-nearest-neighbours
    unweighted_undirected_knn_matrix = undirected_neighbors_graph(data, n_neighbours, mode='connectivity')
    weighted_undirected_knn_matrix = undirected_neighbors_graph(data, n_neighbours, mode='distance')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', numpy.ComplexWarning)
        unweighted_undirected_knn_log_matrix = numpy.array(matrix_logarithm(unweighted_undirected_knn_matrix),
                                                           dtype=float)
        weighted_undirected_knn_log_matrix = numpy.array(matrix_logarithm(weighted_undirected_knn_matrix), dtype=float)

    # Convert distances to similarity
    weighted_undirected_knn_matrix = numpy.asarray(
        [[1 / (1 + x) if x != 0 else 0 for x in column] for column in weighted_undirected_knn_matrix])

    # ============
    # Compute logos
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

        # Spectral logos
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        spectral.fit(data)
        spectral_clustering = spectral.labels_

        # Spectral logos with log matrix
        log_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_spectral.fit(euclidean_distance_log_matrix)
        log_spectral_clustering = log_spectral.labels_

        # Spectral logos with threshold log matrix
        log_spectral_rounded = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_spectral_rounded.fit(euclidean_distance_log_filter_matrix)
        log_spectral_rounded_clustering = log_spectral_rounded.labels_

        # Spectral logos with unweighted undirected kNN matrix
        unweighted_snn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        unweighted_snn_spectral.fit(unweighted_undirected_knn_matrix)
        unweighted_snn_spectral_clustering = unweighted_snn_spectral.labels_

        # Spectral logos with log unweighted undirected kNN matrix
        log_unweighted_snn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_unweighted_snn_spectral.fit(unweighted_undirected_knn_log_matrix)
        log_unweighted_snn_spectral_clustering = log_unweighted_snn_spectral.labels_

        # Spectral logos with weighted undirected kNN matrix
        weighted_snn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        weighted_snn_spectral.fit(weighted_undirected_knn_matrix)
        weighted_snn_spectral_clustering = weighted_snn_spectral.labels_

        # Spectral logos with log weighted undirected kNN matrix
        log_weighted_snn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_weighted_snn_spectral.fit(weighted_undirected_knn_log_matrix)
        log_weighted_snn_spectral_clustering = log_weighted_snn_spectral.labels_

    # Hierarchical logos with weighted undirected kNN matrix
    weighted_snn_agglom = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
    weighted_snn_agglom.fit(1 / (1 + weighted_undirected_knn_matrix))
    weighted_snn_agglom_clustering = weighted_snn_agglom.labels_

    # Hierarchical logos with weighted undirected kNN matrix
    log_agglom = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
    log_agglom.fit(euclidean_distance_log_matrix)
    log_agglom_clustering = log_agglom.labels_

    # SpectACl
    spectacl = Spectacl(n_clusters=n_clusters, epsilon=heuristics.spectacl_eps_heuristic(data))
    spectacl.fit(data)
    spectacl_clustering = spectacl.labels_

    # Inverse Heat Kernel Clustering
    vals, vecs = numpy.linalg.eig(-1 * euclidean_distance_log_matrix)
    vecs = vecs[:, numpy.argsort(vals)]
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(vecs[:, 0:n_clusters])
    inverse_heat_clustering = clusterer.labels_

    clustering_results = (
        ('SC f log euclid', log_spectral_rounded_clustering),
        ('SC log w ud kNN', log_weighted_snn_spectral_clustering),
        ('AC w ud kNN', weighted_snn_agglom_clustering),
        ('AC log w ud kNN', log_agglom_clustering),
        ('IHK SC', inverse_heat_clustering),
        ('k-Means', kmeans_clustering),
        ('DBSCAN', dbscan_clustering),
        ('Spectral\nClustering', spectral_clustering),
        ('SpectACl', spectacl_clustering),
    )

    paper_algorithms = (
        ('Logos', log_spectral_rounded_clustering),
        ('k-Means', kmeans_clustering),
        ('DBSCAN', dbscan_clustering),
        ('Spectral\nClustering', spectral_clustering),
        ('SpectACl', spectacl_clustering),
    )

    normal_results = (
        (None, None),
        (None, None),
        (None, unweighted_snn_spectral_clustering),
        (None, log_weighted_snn_spectral_clustering)
    )

    log_results = (
        ('Euclidean\ndistance', log_spectral_clustering),
        ('Euclidean\ndistance\nwith filter', log_spectral_rounded_clustering),
        ('unweighted\nundirected\nkNN', log_unweighted_snn_spectral_clustering),
        ('weighted\nundirected\nkNN', log_weighted_snn_spectral_clustering)
    )

    if show:
        # figures.compute_and_plot_evaluation(labels, clustering_results)
        figures.compute_and_plot_evaluation(labels, paper_algorithms)
        # plotter.plot_normal_and_log(labels, normal_results, log_results)
        pyplot.show()


if __name__ == '__main__':
    main(show=True)
