def hit_rate_at_k(recommendations, ground_truth, k):
    hits, total = 0, len(ground_truth)

    for recs, true_items in zip(recommendations, ground_truth):
        top_k = recs[:k]

        if any(item in top_k for item in true_items):
            hits += 1

    return hits / total if total > 0 else 0.0