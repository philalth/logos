import numpy
from sklearn.datasets import make_blobs


def make_subspace_blobs(n_samples, n_features, cluster_std, cluster_subspaces, center_box=(-30, 30)):
    """
    Generate isotropic Gaussian subspace blobs for logos.

    Parameters
    ----------
    n_samples: array-like
        Each element of the sequence indicates the number of samples per cluster.

    n_features: int
        The number of total features for each sample.

    cluster_std: array of floats
        The standard deviation of the clusters.

    center_box: pair of floats (min, max)
        The bounding box for each cluster center when centers are generated at random.

    cluster_subspaces:  array-like
        The dimensions of the subspace in which the clusters should be generated.

    Returns
    -------
    data : array of shape [n_samples, n_features]
        The generated samples.

    labels : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    if len(n_samples) != len(cluster_std) or len(n_samples) != len(cluster_subspaces) or len(cluster_std) != len(
            cluster_subspaces):
        raise ValueError('Bad arguments')

    total_n_samples = sum(n_samples)
    n_clusters = len(n_samples)
    data = numpy.random.uniform(center_box[0], center_box[1], size=(total_n_samples, n_features))
    labels = []

    # absolute array starting index of the current cluster
    current_cluster_start_index = 0

    # iterate over all clusters
    for i in range(0, n_clusters):
        samples = n_samples[i]
        std = cluster_std[i]
        subspace = cluster_subspaces[i]

        blob = make_blobs(n_samples=[samples], n_features=len(subspace), cluster_std=std,
                          center_box=center_box, shuffle=False)[0]

        # iterate over all subspace dimensions of the current cluster
        for j in range(0, len(subspace)):
            dimension = subspace[j]

            if dimension >= n_features:
                raise ValueError('Bad arguments')

            # relative array index of the current cluster
            current_index = 0

            # insert clusters into data array
            for k in range(current_cluster_start_index, current_cluster_start_index + samples):
                data[k][dimension] = blob[current_index][j]
                current_index += 1

        current_cluster_start_index += samples
        labels = labels + samples * [i]

    return data, labels
