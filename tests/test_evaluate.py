from docbenchtr.evaluate import compute_metrics


def test_page_level_recall_and_mrr():
    metrics = compute_metrics([8, 3, 5, 2, 9], [3, 5])
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5


def test_recall_for_multiple_relevant_pages():
    metrics = compute_metrics([1, 2, 7, 4, 5], [4, 5])
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.0
    assert metrics["recall@5"] == 1.0
