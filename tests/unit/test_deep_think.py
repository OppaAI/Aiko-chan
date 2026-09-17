from cognition.deep_think import _polarity_signature, flag_contradictions


def test_polarity_signature_matches_complete_terms():
    assert _polarity_signature("I dislike noisy rooms") == {"-like"}
    assert _polarity_signature("I like quiet rooms") == {"+like"}
    assert _polarity_signature("This feature is disabled") == set()
    assert _polarity_signature("Please disable this feature") == {"-enable"}


def test_polarity_signature_negation_phrase_counts_once():
    # "don't want" must not also count as a positive "want" — otherwise two
    # agreeing memories flag each other as contradictions.
    assert _polarity_signature("I don't want extra notifications") == {"-want"}
    assert _polarity_signature("I want extra notifications") == {"+want"}


def test_flag_contradictions_ignores_agreeing_negations():
    rows = flag_contradictions([
        {"memory": "I don't want extra notifications please"},
        {"memory": "I don't want extra notifications ever"},
    ])
    assert not any(m.get("_contradiction") for m in rows)


def test_flag_contradictions_still_catches_real_opposition():
    rows = flag_contradictions([
        {"memory": "I don't want extra notifications"},
        {"memory": "I want extra notifications now"},
    ])
    assert all(m.get("_contradiction") for m in rows)
