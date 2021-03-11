import numpy
from scipy.io.arff import loadarff
from sklearn.neighbors import kneighbors_graph


def load_arff(path):
    dataset = loadarff(path)
    data = numpy.array(dataset[0])
    data = [list(i) for i in data]
    # Spaghetti Code um nach dem letzten Wert (Label) zu sortieren
    data = numpy.array(data)
    data.sort(axis=0)

    res = numpy.zeros((len(data), 2))
    labels = []
    k = 0
    bla = []
    for i in data:
        res[k] = i[0:len(i) - 1]
        bla.append(i[0:len(i)])
        labels.append(int(i[len(i) - 1]))
        k = k + 1
    return res, labels


def add_noise(data, percentage):
    """

    @param data: data as a tuple of (datapoints, labels)
    @param percentage: percentage compared to the total size of the data of the number of noise to be added
    @return: (datapoints, labels) as tuple with noise points and labels added
    """
    datapoints, labels = data
    n_noise = int(len(datapoints) * percentage)

    noise = numpy.random.uniform(-50, 50, size=(n_noise, 2))

    datapoints = numpy.concatenate([datapoints, noise])
    labels = numpy.concatenate([labels, [-1] * len(noise)])

    return datapoints, labels


def undirected_neighbors_graph(data, k, mode='connectivity'):
    knn_graph = kneighbors_graph(data, k, mode=mode).toarray()
    snn_graph = numpy.zeros((len(knn_graph), len(knn_graph)))

    for i in range(0, len(knn_graph)):
        for j in range(i, len(knn_graph)):
            value = max(knn_graph[i][j], knn_graph[j][i])
            snn_graph[i][j], snn_graph[j][i] = value, value

    return snn_graph
