import warnings
from itertools import islice, cycle

import numpy
from matplotlib import pyplot
from sklearn.metrics import normalized_mutual_info_score


def plot_matrices_and_logarithm(matrices):
    fig = pyplot.figure(figsize=[20, 8])
    plot_num = 1
    for matrix, log_matrix, name in matrices:
        # plot original matrix
        fig.add_subplot(1, len(matrices), plot_num)
        pyplot.imshow(matrix)
        pyplot.colorbar()
        pyplot.title(name)

        # plot log matrix
        fig.add_subplot(2, len(matrices), plot_num)
        log_matrix = numpy.ma.masked_where(log_matrix == 0, log_matrix)
        colormap = pyplot.get_cmap('rainbow')
        colormap.set_bad(color='black')
        pyplot.imshow(log_matrix, cmap=colormap)
        pyplot.colorbar()
        pyplot.title(name + ' logarithm')

        plot_num += 1


def plot_data_and_matrices(data, distance_matrix, distance_matrix_log, description=''):
    datapoints, labels = data

    fig = pyplot.figure(figsize=[20, 8])

    fig.add_subplot(1, 3, 1)
    pyplot.scatter(datapoints[:, 0], datapoints[:, 1], c=labels, s=14)

    pyplot.xlabel('1st dimension')
    pyplot.ylabel('2nd dimension')
    pyplot.title('Data: ' + description)

    fig.add_subplot(1, 3, 2)
    pyplot.imshow(distance_matrix)
    pyplot.colorbar()
    pyplot.title('Affinity matrix')

    fig.add_subplot(1, 3, 3)
    a = distance_matrix_log
    a = numpy.ma.masked_where(a == 0, a)
    colormap = pyplot.get_cmap('rainbow')
    colormap.set_bad(color='black')
    pyplot.imshow(a, cmap=colormap)
    pyplot.colorbar()
    pyplot.title('Affinity matrix logarithm')


def plot_clustering(algorithms, data):
    """
    Plots logos results. If the data has more than two dimensions
    it will be projected to the first two dimensions.

    Parameters
    ----------
    algorithms: (cluster labels, name) tuple

    data: array
        The training data.
    """
    fig = pyplot.figure(figsize=[20, 8])
    plot_num = 1
    for name, clustering in algorithms:
        colors = numpy.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                         int(max(clustering) + 1))))
        # add black color for outliers (if any)
        colors = numpy.append(colors, ["#000000"])

        fig.add_subplot(1, len(algorithms), plot_num)
        pyplot.scatter(data[:, 0], data[:, 1], color=colors[clustering], s=7)
        pyplot.title(name)
        plot_num += 1


def compute_and_plot_evaluation(true_labels, algorithms):
    # ============
    # Calculate evaluation scores
    # ============

    bar_labels = []
    nmi_scores = []

    for name, clustering in algorithms:
        bar_labels.append(name)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            nmi_score = normalized_mutual_info_score(true_labels, clustering)

            # current version of sklearn's NMI implementation is bugged if all labels are the same i.e. all
            # datapoints are labeled as one cluster, in this case the NMI exceeds its defined range of [0.0, 1.0]
            if 0 <= nmi_score <= 1:
                nmi_scores.append(nmi_score)
            else:
                nmi_scores.append(0.0)

    # ============
    # Plot evaluation scores
    # ============

    pyplot.subplots()
    index = numpy.arange(len(algorithms))
    bar_width = 0.2
    opacity = 1

    pyplot.bar(index, nmi_scores, bar_width,
               alpha=opacity,
               color='#ff7f00')

    pyplot.ylim(0, 1.1)
    pyplot.ylabel('NMI')
    pyplot.xticks(index, bar_labels, rotation=45, ha='right')

    pyplot.tight_layout()


def plot_normal_and_log(true_labels, algorithms, log_algorithms):
    # ============
    # Calculate evaluation scores
    # ============

    bar_labels = []
    nmi_scores = []
    log_nmi_scores = []

    for name, log_clustering in log_algorithms:
        bar_labels.append(name)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            log_nmi_score = normalized_mutual_info_score(true_labels, log_clustering)

            if 0 <= log_nmi_score <= 1:
                log_nmi_scores.append(log_nmi_score)
            else:
                log_nmi_scores.append(0.0)

    for name, clustering in algorithms:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            if clustering is None:
                nmi_score = 0.0
            else:
                nmi_score = normalized_mutual_info_score(true_labels, clustering)

            # current version of sklearn's NMI implementation is bugged if all labels are the same i.e. all
            # datapoints are labeled as one cluster, in this case the NMI exceeds its defined range of [0.0, 1.0]
            if 0 <= nmi_score <= 1:
                nmi_scores.append(nmi_score)
            else:
                nmi_scores.append(0.0)

    # ============
    # Plot evaluation scores
    # ============

    pyplot.subplots()
    index = numpy.arange(len(algorithms))
    bar_width = 0.2
    opacity = 1

    pyplot.bar(index, nmi_scores, bar_width,
               alpha=opacity,
               color='#377eb8',
               label='without log')

    pyplot.bar(index + bar_width, log_nmi_scores, bar_width,
               alpha=opacity,
               color='#ff7f00',
               label='with log')

    pyplot.ylim(0, 1.1)
    pyplot.ylabel('NMI')
    pyplot.xticks(index + bar_width / 2, bar_labels, rotation=45, ha='center')

    pyplot.legend()
    pyplot.tight_layout()
