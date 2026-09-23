"""Unit tests for the motion director (conversation -> gesture mapping)."""
from interface.webui.motion_director import MotionDirector, KNOWN_GESTURES


def _director():
    d = MotionDirector()
    d.reset_cooldowns()
    return d


def test_praise_triggers_clap():
    d = _director()
    assert d.suggest("You're amazing, great job!", emotion="excited",
                     intensity=0.8) == "clap"


def test_thanks_triggers_bow():
    d = _director()
    assert d.suggest("thank you so much", emotion="warm",
                     intensity=0.5) == "bow"


def test_apology_triggers_hands_clasp():
    d = _director()
    assert d.suggest("sorry about that", emotion="neutral",
                     intensity=0.4) == "handsClasp"


def test_greeting_and_farewell_wave():
    d = _director()
    assert d.suggest("hello Aiko!", emotion="warm",
                     intensity=0.5) == "wave"
    d.reset_cooldowns()
    assert d.suggest("goodnight, see you tomorrow", emotion="calm",
                     intensity=0.4) == "wave"


def test_celebration_triggers_dance():
    d = _director()
    assert d.suggest("we did it! let's dance, it's my birthday!",
                     emotion="excited", intensity=0.9) == "dance"


def test_playful_triggers_giggle():
    d = _director()
    assert d.suggest("hehe that was funny", emotion="playful",
                     intensity=0.7) == "giggle"


def test_lean_in_impulse():
    d = _director()
    assert d.suggest("I'm feeling a bit down today", emotion="sad",
                     intensity=0.5,
                     impulse="soften_and_lean_in") == "leanIn"


def test_curious_triggers_tilt():
    d = _director()
    assert d.suggest("what do you think about black holes?",
                     emotion="curious", intensity=0.6) == "curiousTilt"


def test_global_cooldown_suppresses_spam():
    d = _director()
    first = d.suggest("hello Aiko!", emotion="warm", intensity=0.5)
    assert first == "wave"
    # immediate second suggestion suppressed by the global cooldown
    assert d.suggest("thank you!", emotion="warm", intensity=0.5) is None


def test_per_gesture_cooldown():
    d = _director()
    d.suggest("hello Aiko!", emotion="warm", intensity=0.5)
    d._last_any_t = 0.0  # clear only the global cooldown
    # same gesture still cooling down
    assert d.suggest("hey Aiko, hello again", emotion="warm",
                     intensity=0.5) is None


def test_suggested_gestures_are_known():
    assert {"clap", "dance", "wave", "giggle", "bow"} <= KNOWN_GESTURES


def test_on_listening_start():
    assert MotionDirector().on_listening_start() == "leanIn"


def test_boring_turn_suggests_nothing():
    d = _director()
    assert d.suggest("the meeting is at 3pm", emotion="neutral",
                     intensity=0.3) is None


def test_cues_only_match_at_word_boundaries():
    d = _director()
    assert d.suggest("this technical partition describes a musician") is None
