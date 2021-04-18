import numpy
import pandas
from sklearn.datasets import make_circles, make_moons, make_blobs


def load_two_circles():
    """Loads the two circles dataset, a simple toy dataset to visualize
    clustering and classification algorithms.

    Returns
    -------
    X : ndarray of shape (1500, 2)
        The generated samples.

    y : ndarray of shape (1500,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    return make_circles(n_samples=1500, factor=.5, noise=.05, shuffle=False)


def load_two_moons():
    """Loads the two moons dataset, a simple toy dataset to visualize
    clustering and classification algorithms.

    Returns
    -------
    X : ndarray of shape (1500, 2)
        The generated samples.

    y : ndarray of shape (1500,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    return make_moons(n_samples=1500, noise=.05, shuffle=False)


def load_anisotropic_blobs():
    """Generate anisotropic Gaussian blobs for clustering.

    Returns
    -------
    X : ndarray of shape (1500, 2)
        The generated samples.

    y : ndarray of shape (1500,)
        The integer labels for cluster membership of each sample.
    """
    datapoints, labels = make_blobs(n_samples=1500, random_state=170,
                                    shuffle=False)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    datapoints = numpy.dot(datapoints, transformation)
    return datapoints, labels


def load_varied_variances_blobs():
    """Generate isotropic Gaussian blobs with varied variances for clustering.

    Returns
    -------
    X : ndarray of shape (1500, 2)
        The generated samples.

    y : ndarray of shape (1500,)
        The integer labels for cluster membership of each sample.
    """
    return make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5],
                      random_state=170)


def load_pulsar():
    """
    Load and return the T4-8K data set.

    =================   ==============
    Classes                          6
    Samples per class            ~1000
    Samples total                 8000
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Source: .https://archive.ics.uci.edu/ml/datasets/HTRU2.

    Returns
    -------
    data : ndarray
        Pulsar data set.
    """
    data = pandas.read_csv('datasets/pulsar.csv').to_numpy()
    datapoints = numpy.array(data[:, 0: 8], dtype=float)
    labels = numpy.array(data[:, 8], dtype=int)
    return datapoints, labels


def load_mouse():
    """
    Load and return the mouse data set, consisting of three Gaussian
    clusters which form head and ears of Micky Mouse.

    =================   ==============
    Classes                          3
    Samples per class             ~100
    Samples total                  500
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Source: https://github.com/elki-project/elki/blob/master/data/synthetic/Vorlesung/mouse.csv.

    Returns
    -------
    data : ndarray
        Mouse data set.
    """
    data = pandas.read_csv('datasets/mouse.csv').to_numpy()
    datapoints = numpy.array(data[:, 0: 2], dtype=float)
    labels = numpy.array(data[:, 2], dtype=int)
    return datapoints, labels


def load_gaussian_clusters():
    """
    Load and return synthetic Gaussian clusters with different
    degree of cluster overlap.

    =================   ==============
    Classes                         15
    Samples per class             ~330
    Samples total                 5000
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Source: http://cs.joensuu.fi/sipu/datasets/.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are.
    """
    f = open('datasets/s1.txt', 'r')

    data = []
    for line in f.readlines():
        data.append(line.replace('\n', '').split('    ')[1:])

    f.close()
    return numpy.array(data, dtype=float)


def load_spirals():
    """
    Load and return synthetic spirals.

    =================   ==============
    Classes                          3
    Samples per class              104
    Samples total                  312
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Source: http://cs.joensuu.fi/sipu/datasets/.

    Returns
    -------
    data : Bunch
        Dictionary-like object,
    """
    data = []
    labels = []

    f = open('datasets/spiral.txt', 'r')
    for line in f.readlines():
        data.append(line.replace('\n', '').split('\t')[:2])
        labels.append(line.replace('\n', '').split('\t')[2])
    f.close()

    return numpy.array(data, dtype=float), numpy.array(labels, dtype=int)


def load_t48k():
    """
    Load and return the T4-8K data set.

    =================   ==============
    Classes                          6
    Samples per class            ~1000
    Samples total                 8000
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Source: http://cs.joensuu.fi/sipu/datasets/.

    Returns
    -------
    data : ndarray
        T4-8K data set.
    """
    data = []

    f = open('datasets/t4.8k.txt', 'r')
    for line in f.readlines():
        data.append(line.replace('\n', '').split(' '))
    f.close()

    return numpy.array(data, dtype=float)
