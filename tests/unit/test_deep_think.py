from cognition.deep_think import _polarity_signature


def test_polarity_signature_matches_complete_terms():
    assert _polarity_signature("I dislike noisy rooms") == {"-like"}
    assert _polarity_signature("I like quiet rooms") == {"+like"}
    assert _polarity_signature("This feature is disabled") == set()
    assert _polarity_signature("Please disable this feature") == {"-enable"}
