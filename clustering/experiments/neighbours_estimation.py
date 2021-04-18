import sys
import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import normalized_mutual_info_score

from utils.data_utils import undirected_neighbors_graph
from utils.matrix_logarithm import matrix_logarithm

# Create data
data, labels = fetch_olivetti_faces(return_X_y=True)
n_clusters = 40

neighbour_values = range(5, 200, 1)

# Initialize NMI scores
knn_results, weighted_undirected_knn_results, log_weighted_undirected_knn_results = [], [], []

for n_neighbours in neighbour_values:
    sys.stdout.write("\rIteration: %f" % n_neighbours)
    sys.stdout.flush()

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        # Spectral Clustering with kNN
        knn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbours)
        knn_spectral.fit(data)
        knn_clustering = knn_spectral.labels_

        # Spectral Clustering with weighted undirected kNN matrix
        weighted_undirected_knn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        weighted_undirected_knn_spectral.fit(weighted_undirected_knn_matrix)
        weighted_undirected_knn_clustering = weighted_undirected_knn_spectral.labels_

        # Spectral Clustering with weighted undirected kNN log matrix
        log_weighted_undirected_knn_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_weighted_undirected_knn_spectral.fit(weighted_undirected_knn_log_matrix)
        log_weighted_undirected_knn_clustering = log_weighted_undirected_knn_spectral.labels_

    # Compute NMI scores
    knn_nmi = normalized_mutual_info_score(labels, knn_clustering)
    weighted_undirected_knn_nmi = normalized_mutual_info_score(labels, weighted_undirected_knn_clustering)
    log_weighted_undirected_knn_nmi = normalized_mutual_info_score(labels, log_weighted_undirected_knn_clustering)

    knn_results.append(knn_nmi)
    weighted_undirected_knn_results.append(weighted_undirected_knn_nmi)
    log_weighted_undirected_knn_results.append(log_weighted_undirected_knn_nmi)

# Plot results as line plot
pyplot.xlabel('Number of nearest neighbours')
pyplot.ylabel('NMI')
pyplot.ylim(-0.05, 1.05)

pyplot.plot(neighbour_values, knn_results, label='kNN', linestyle='-')
pyplot.plot(neighbour_values, weighted_undirected_knn_results, label='w ud kNN', linestyle='--')
pyplot.plot(neighbour_values, log_weighted_undirected_knn_results, label='w ud kNN log', linestyle='-.')

pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Neighbours.png", bbox_inches='tight')
pyplot.show()
