import numpy as np


def rank_positions(score_dict):
    ranked = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    return {page: idx + 1 for idx, (page, _) in enumerate(ranked)}


def spearman_rank_correlation(rank_a, rank_b):
    pages = sorted(rank_a.keys())
    n = len(pages)
    d2_sum = 0.0
    for page in pages:
        d = rank_a[page] - rank_b[page]
        d2_sum += d * d
    return 1.0 - (6.0 * d2_sum) / (n * (n * n - 1))


def pearson_score_correlation(score_a, score_b):
    pages = sorted(score_a.keys())
    a = np.array([score_a[p] for p in pages], dtype=float)
    b = np.array([score_b[p] for p in pages], dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def inversion_count(rank_a, rank_b):
    pages = sorted(rank_a.keys())
    count = 0
    n = len(pages)
    for i in range(n):
        for j in range(i + 1, n):
            p = pages[i]
            q = pages[j]
            order_a = rank_a[p] - rank_a[q]
            order_b = rank_b[p] - rank_b[q]
            if order_a * order_b < 0:
                count += 1
    return count

