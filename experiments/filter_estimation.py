import sys
import warnings

import numpy
from matplotlib import pyplot
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances

from utils.matrix_logarithm import matrix_logarithm

# Create data

data, labels = fetch_olivetti_faces(return_X_y=True)
n_clusters = 40

filter_values = numpy.arange(0.0, 0.31, 0.01)

# Initialize NMI scores
euclidean_log_filter_results = []

euclidean_distance_matrix = euclidean_distances(data, data)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', numpy.ComplexWarning)
    euclidean_distance_log_matrix = numpy.array(matrix_logarithm(euclidean_distance_matrix), dtype=float)

# Spectral clustering with log matrix
log_spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
log_spectral.fit(euclidean_distance_log_matrix)
log_spectral_clustering = log_spectral.labels_

for filter_value in filter_values:
    sys.stdout.write("\rIteration: %f" % filter_value)
    sys.stdout.flush()

    euclidean_distance_log_filter_matrix = numpy.asarray(
        [[x if x > filter_value else 0 for x in column] for column in euclidean_distance_log_matrix])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        # Spectral clustering with threshold log matrix
        log_spectral_rounded = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        log_spectral_rounded.fit(euclidean_distance_log_filter_matrix)
        log_spectral_rounded_clustering = log_spectral_rounded.labels_

    euclidean_log_filter_results.append(normalized_mutual_info_score(labels, log_spectral_rounded_clustering))

pyplot.xlabel('Threshold value')
pyplot.ylabel('NMI')
pyplot.ylim(-0.05, 1.05)
pyplot.plot(filter_values, [normalized_mutual_info_score(labels, log_spectral_clustering)] * len(filter_values),
            label='without threshold', linestyle='--')
pyplot.plot(filter_values, euclidean_log_filter_results, label='with threshold')

pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("test.png", bbox_inches='tight')
pyplot.show()

print('\n' + str(numpy.argmax(euclidean_log_filter_results)))
