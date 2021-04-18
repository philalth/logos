import itertools

from matplotlib import pyplot


def evaluation_line_plot(results, title='', x_axis_title='', x_range=None):
    """
    Creates a line plot for the results of clustering algorithms.

    Parameters
    ----------
    results: ((scores, name), ...)
        Results to be plotted as a tuple of tuples consisting of the results
        as a list and the name of the used algorithm.

    title: str
        Text to use for the title.

    x_axis_title: str
        The label text for the x-axis.

    x_range: array of ints

    """
    marker = itertools.cycle(('o', '^', 's', 'd'))
    linestyle = itertools.cycle((':', '-.', '-'))

    pyplot.title(title)
    pyplot.xlabel(x_axis_title)
    pyplot.ylabel('NMI')
    pyplot.ylim(-0.1, 1.1)

    for scores, algorithm in results:
        pyplot.plot(x_range, scores, label=algorithm, linestyle=next(linestyle), marker=next(marker))

    pyplot.legend()
    pyplot.show()
