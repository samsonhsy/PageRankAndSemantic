import json
import numpy as np

def load_links(path):
    with open(path, "r", encoding="utf-8") as f:
        links = json.load(f)  # json to dict
    return links


def build_H_from_links_json(path):
    links = load_links(path)
    pages = sorted(links.keys())

    idx = {p: i for i, p in enumerate(pages)}
    n = len(pages)
    H = np.zeros((n, n), dtype=float)
    for page in pages:
        i = idx[page]
        outs = links[page]
        out_degree = len(outs)
        if out_degree == 0:
            continue  # dangling row stays all zeros in H
        prob = 1.0 / out_degree
        for out in outs:
            j = idx[out]
            H[i, j] = prob

    # print("H:\n", H)
    return pages, H

def build_H_tilde(H):
    n = H.shape[0]
    row_sums = H.sum(axis=1) # row wise sum

    w = np.zeros((n, 1), dtype=float)
    for i in range(n):
        if row_sums[i] == 0:
            w[i, 0] = 1.0
        else:
            w[i, 0] = 0.0
    H_tilde = H + (w @ np.ones((1, n))) / n

    # print("H_tilde:\n", H_tilde)

    return H_tilde

def build_G(H_tilde):
    n = H_tilde.shape[0]
    theta = 0.85
    arr = np.full((n, n), 1 / n)
    G = theta * H_tilde + (1-theta) * arr
    # print("G:\n", G)

    return G

def pagerank_iterative(G, epsilon=1e-6, max_iter=100):
    n = G.shape[0]
    pi = np.ones(n) / n  # pi^T[0]
    pi_history = [pi.copy()]
    for t in range(1, max_iter + 1):
        pi_next = pi @ G  # pi^T[t] = pi^T[t-1] G
        diff = np.linalg.norm(pi_next - pi, ord=1)
        pi_history.append(pi_next.copy())
        pi = pi_next
        if diff < epsilon:
            return pi, t, diff, pi_history
    return pi, max_iter, diff, pi_history


def compute_pagerank(links_path, epsilon=1e-6, max_iter=100):
    pages, H = build_H_from_links_json(links_path)
    H_tilde = build_H_tilde(H)
    G = build_G(H_tilde)
    pi, t, diff, pi_history = pagerank_iterative(G, epsilon=epsilon, max_iter=max_iter)
    score_dict = {pages[i]: float(pi[i]) for i in range(len(pages))}
    return score_dict, t, diff, pi_history


def rank_scores(score_dict):
    return sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

if __name__ == "__main__":
    scores, t, diff, _ = compute_pagerank("data/links.json")
    print("Converged in iterations:", t)
    print("Final diff:", diff)
    print("PageRank scores:")
    for page, score in rank_scores(scores):
        print(page, score)
