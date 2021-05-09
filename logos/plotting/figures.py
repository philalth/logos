import warnings

import numpy
from matplotlib import pyplot
from sklearn.metrics import normalized_mutual_info_score


def show_matrix(matrix, filename='matrix'):
    matrix = numpy.ma.masked_where(matrix == 0, matrix)
    colormap = pyplot.get_cmap('rainbow')
    colormap.set_bad(color='black')
    pyplot.imshow(matrix, cmap=colormap)
    pyplot.colorbar()
    pyplot.savefig(filename, bbox_inches='tight')
    pyplot.show()


def show_clustering(data, clustering, filename='logos'):
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis('equal')
    pyplot.scatter(data[:, 0], data[:, 1], c=clustering, s=7)
    pyplot.savefig(filename, bbox_inches='tight')
    pyplot.show()


def compute_and_plot_evaluation(true_labels, algorithms):
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

    pyplot.subplots()
    index = numpy.arange(len(algorithms))
    bar_width = 0.2
    opacity = 1

    print(nmi_scores)

    pyplot.bar(index, nmi_scores, bar_width,
               alpha=opacity)
               # color=['purple', 'purple', 'olive', 'olive', 'blue', 'brown', 'brown', 'brown', 'brown'])

    pyplot.ylim(0, 1.05)
    pyplot.ylabel('NMI')
    pyplot.xticks(index, bar_labels, rotation=45, ha='right')

    pyplot.tight_layout()
    pyplot.savefig("test.png", bbox_inches='tight')
    pyplot.show()
