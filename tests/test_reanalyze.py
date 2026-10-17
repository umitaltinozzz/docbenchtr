from docbenchtr.reanalyze import _sign_test_p, score_ranking_groups


def test_equivalent_pages_are_alternatives_for_one_provision():
    groups = [[21, 139], [22]]

    one_provision = score_ranking_groups([139], groups)
    both_provisions = score_ranking_groups([139, 22], groups)

    assert one_provision["recall@5"] == 0.5
    assert both_provisions["recall@5"] == 1.0
    assert both_provisions["reciprocal_rank"] == 1.0


def test_sign_test_is_exact_and_two_sided():
    assert _sign_test_p(8, 3) == 0.2265625
    assert _sign_test_p(3, 8) == 0.2265625
