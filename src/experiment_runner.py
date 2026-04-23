from pathlib import Path

from hybrid_ranking import hybrid_scores, rank_scores
from metrics import inversion_count, pearson_score_correlation, rank_positions, spearman_rank_correlation
from pagerank_ranking import compute_pagerank
from semantic_ranking import semantic_scores


QUERIES = [
    "machine learning basics",
    "supervised vs unsupervised learning",
    "computer science fundamentals",
    "algorithms and data structures basics",
    "applications of computer science",
]

HYBRID_ALPHA = 0.5


def print_ranking_block(title, score_dict):
    print(title)
    for i, (page, score) in enumerate(rank_scores(score_dict), start=1):
        print(f"  {i}. {page:<12} {score:.6f}")


def run_experiment():
    links_path = "./data/links.json"

    pagerank_scores, iters, diff, _ = compute_pagerank(str(links_path))
    print("=== Global PageRank ===")
    print(f"Converged in {iters} iterations, final diff={diff:.8f}")
    print_ranking_block("PageRank ranking:", pagerank_scores)

    print("\n=== Query Experiments ===")
    for query in QUERIES:
        sem_scores = semantic_scores(query, cache_path="./data/html_embedding.json")

        hy_scores = hybrid_scores(pagerank_scores, sem_scores, alpha=HYBRID_ALPHA)

        rank_pr = rank_positions(pagerank_scores)
        rank_sem = rank_positions(sem_scores)

        spearman = spearman_rank_correlation(rank_pr, rank_sem)
        pearson = pearson_score_correlation(pagerank_scores, sem_scores)
        inv = inversion_count(rank_pr, rank_sem)

        print(f"--- Query: {query} ---")
        print_ranking_block("Semantic ranking:", sem_scores)
        print_ranking_block(f"Hybrid ranking (alpha={HYBRID_ALPHA}):", hy_scores)
        print(
            "Metrics PageRank vs Semantic: "
            f"Spearman={spearman:.4f}, Pearson={pearson:.4f}, "
            f"Inversions={inv}"
        )
        print("\n")


if __name__ == "__main__":
    run_experiment()
