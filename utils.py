"""
Source: https://github.com/qbarthelemy/PyPermut/blob/main/examples/compute_auroc_pvalue.py

"""
import numpy as np
from scipy.stats import percentileofscore


def permutation_metric(y_true, y_score, func, *, n=10000, side='right'):
    """Permutation test for machine learning metric.

    This function performs a permutation test on any metric based on the
    predictions of a model. It permutes labels and predictions to obtain a
    p-value for any machine learning metrics:

    * the Area Under the Receiver Operating Characteristic (AUROC) curve,
    * the Area Under the Precision-Recall (AUPR) curve,
    * the negative log-likelihood (log-loss),
    * etc.

    Parameters
    ----------
    y_true : array_like, shape (n_samples, n_classes)
        True binary labels, with first dimension representing the sample
        dimension and with second dimension representing the different classes.

    y_score : array_like, shape (n_samples, n_classes)
        Scores of prediction, same dimensions as y_true. Scores can be
        probabilities or labels.

    func : callable
        Function to compute the metric, with signature `func(y_true, y_score)`.

    n : int (default 10000)
        Number of permutations for the permutation test.

    side : string (default 'right')
        Side of the test:

        * 'left' for a left-sided test,
        * 'two' or 'double' for a two-sided test,
        * 'right' for a right-sided test.

    Returns
    -------
    m : float
        The value of the metric.

    pval : float
        The p-value associated to the metric.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            'Inputs y_true and y_score do not have compatible dimensions: '
            'y_true is of dimension {} while y_score is {}.'
            .format(y_true.shape, y_score.shape))
    n_samples = y_true.shape[0]

    # under the null hypothesis, sample the metric distribution
    null_dist = np.empty(n, dtype=float)
    for p in range(n):
        permuted_indices = np.random.permutation(n_samples)
        null_dist[p] = func(y_true[permuted_indices], y_score)

    # compute the real metric
    m = func(y_true, y_score)
    perc = percentileofscore(null_dist, m, kind='strict')
    pval = perc_to_pval(perc, side)

    return m, pval


def perc_to_pval(perc, side):
    """Transform percentile into p-value, depending on the side of the test.

    Parameters
    ----------
    perc : float
        Percentile of the observed statistic, in [0, 100].

    side : string
        Side of the test:

        * 'left', 'lower' or 'less', for a left-sided test;
        * 'two', 'double' or 'two-sided', for a two-sided test;
        * 'right', 'upper' or 'greater', for a right-sided test.

    Returns
    -------
    pval : float
        The p-value associated to the stat.
    """
    if not 0 <= perc <= 100:
        raise ValueError('Input percentile="{}" must be included in [0, 100].'
                         .format(perc))

    if side in ['left', 'lower', 'less']:
        pval = perc / 100
    elif side in ['two', 'double', 'two-sided']:
        pval = 2 * min(perc / 100, (100 - perc) / 100)
    elif side in ['right', 'upper', 'greater']:
        pval = (100 - perc) / 100
    else:
        raise ValueError('Invalid value for side="{}".'.format(side))

    return pval
