import copy
import math

import numpy as np
import scipy.stats
from scipy.stats import stats


def concordant_pair_ratio(list1, list2):
    """Proposed `cpr`"""
    assert len(list1) == len(list2)
    total_number = len(list1)
    num_concordant = 0
    for i in range(len(list1)):
        if list1[i] * list2[i] > 0:
            num_concordant += 1
    res = num_concordant / (total_number + 1e-9)
    return res


def pearson(true_vector, pred_vector):
    n = len(true_vector)
    # simple sums
    sum1 = sum(float(true_vector[i]) for i in range(n))
    sum2 = sum(float(pred_vector[i]) for i in range(n))
    # sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in true_vector])
    sum2_pow = sum([pow(v, 2.0) for v in pred_vector])
    # sum up the products
    p_sum = sum([true_vector[i] * pred_vector[i] for i in range(n)])
    # 分子num，分母den
    num = p_sum - (sum1 * sum2 / n)
    try:
        den = math.sqrt((sum1_pow - pow(sum1, 2) / n) *
                        (sum2_pow - pow(sum2, 2) / n) + 1e-8)
    except ValueError:
        return 0

    if den == 0:
        return 0.0
    return num / den


def kendalltau(true_vector, pred_vector):
    tau, p_value = scipy.stats.kendalltau(true_vector, pred_vector, nan_policy='omit')
    return tau


def spearman(true_vector, pred_vector):
    coef, p_value = scipy.stats.spearmanr(true_vector, pred_vector, nan_policy='omit')
    return coef
