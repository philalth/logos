import numpy
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors


def _sorted_knn_distances(data, k):
    """
    Computes the kNN distances and sorts them in ascending order.

    Parameters
    ----------
    data: array
        The training data.
    k: int
        Number of neighbours to get.

    Returns
    -------
    distances: array
        Ordered kNN distances.
    """
    neighbours = NearestNeighbors(n_neighbors=k).fit(data)

    distances, indices = neighbours.kneighbors(data)
    distances = numpy.sort(distances, axis=0)
    distances = distances[:, 1]

    return distances


def dbscan_eps_heuristic(data, k):
    """
    Plots a sorted kNN distance plot in order to estimate the eps
    parameter for DBSCAN.

    References:
    Schubert, Erich, et al. "DBSCAN revisited, revisited: why and how
    you should (still) use DBSCAN."
    ACM Transactions on Database Systems (TODS) 42.3 (2017): 19.

    Parameters
    ----------
    data: array
        The training data.
    k: int
        Number of neighbours to get.
    """
    pyplot.plot(_sorted_knn_distances(data, k))
    pyplot.show()
    eps = float(input("Estimated epsilon value: "))
    return eps


def spectacl_eps_heuristic(data):
    """
    Heuristic used for the eps parameter of the SpectACl clustering algorithm
    taken from the original paper. The value is computed such that 90% of the
    data have at least 10 neighbours.

    References:
    Hess, Sibylle, et al. "The SpectACl of nonconvex clustering: A spectral
    approach to density-based clustering." Proceedings of the AAAI Conference
    on Artificial Intelligence. Vol. 33. 2019.

    Parameters
    ----------
    data: array
        The training data.

    Returns
    -------
    Heuristic eps value.
    """
    n_samples = len(data)
    knn_distances = _sorted_knn_distances(data, 10)
    return knn_distances[int(0.9 * n_samples)]
