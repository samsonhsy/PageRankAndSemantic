def minmax_normalize(score_dict):
    values = list(score_dict.values())
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {k: 0.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}


def hybrid_scores(pagerank_scores, semantic_scores, alpha=0.5):
    pr_norm = minmax_normalize(pagerank_scores)
    sem_norm = minmax_normalize(semantic_scores)
    pages = sorted(pagerank_scores.keys())
    return {
        page: alpha * pr_norm[page] + (1.0 - alpha) * sem_norm[page]
        for page in pages
    }


def rank_scores(score_dict):
    return sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
