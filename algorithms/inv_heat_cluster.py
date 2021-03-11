from sklearn.cluster import KMeans
import numpy


class InverseHeatClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.__check_params__()

    def __check_params__(self):
        """
        Checks, whether the class instance has been parametrized correctly.
        """
        supported_affinities = ["radius_neighbors", "precomputed"]
        if self.n_clusters < 2:
            raise ValueError("Illegal number of clusters specified: {0}".format(str(self.n_clusters)))

    def fit(self, X):
        """
        Clusters the dataset or the precomputed adjacency matrix using the SpectAcl method.
        :param X: The dataset or the adjacency matrix to use.
        """
        eigenvalues, eigenvectors = numpy.linalg.eig(-1 * X)
        eigenvectors = eigenvectors[:, numpy.argsort(eigenvalues)]
        base_labels = KMeans(n_clusters=self.n_clusters).fit_predict(eigenvectors[:, 1:self.n_clusters])

        self.labels_ = base_labels

    def fit_predict(self, X):
        """
        Clusters the dataset or the precomputed adjacency matrix using the SpectAcl method.

        In addition to the `fit` method, the cluster assignment vector is returned.

        :param X: The dataset or the adjacency matrix to use.
        :return: A vector of labels, representing the cluster assignment for each point.
        """
        self.fit(X)
        return self.labels_
