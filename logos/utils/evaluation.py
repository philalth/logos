import warnings

from sklearn.metrics import normalized_mutual_info_score


def nmi_scores(results, ground_truth_labels):
    scores = []

    for name, result in results:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            nmi = normalized_mutual_info_score(ground_truth_labels, result)

        # current version of sklearn's NMI implementation is broken if all
        # labels are the same i.e. all datapoints are labeled as one cluster,
        # in this case the NMI exceeds its defined range of [0, 1]
        if nmi < 0 or nmi > 1:
            nmi = 0.0

        scores.append(nmi)

    return scores
