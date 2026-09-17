from app.retrieval.hybrid import KeywordIndex, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_consistent_results() -> None:
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "d", "a"]],
    )
    assert fused[0] == "a" or fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_keyword_index_returns_matching_document() -> None:
    index = KeywordIndex()
    index.build(
        [
            ("doc-1", "refund policy for enterprise customers"),
            ("doc-2", "deployment guide for internal services"),
        ]
    )
    results = index.search("refund policy")
    assert results[0] == "doc-1"
