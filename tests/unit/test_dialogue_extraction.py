"""
test_dialogue_extraction.py
Tests for extracting spoken dialogue from Aiko messages containing emoji headers and non-verbal cues.
"""

import unittest
from sensory.speak import extract_dialogue_for_tts, format_for_display, parse_aiko_response


class TestDialogueExtraction(unittest.TestCase):

    def test_emoji_header_and_non_verbal(self):
        text = "😊: *sighs softly* (inner thoughts: glad he asked) I'm doing well, thank you."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "sighs softly I'm doing well, thank you.")

    def test_angry_emoji_and_actions(self):
        text = "😒: *crosses arms* You really don't know? *looks away* Let me explain."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "crosses arms You really don't know? looks away Let me explain.")

    def test_bracketed_feelings(self):
        text = "🤖: [analyzing parameters...] Here are the results."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "Here are the results.")

    def test_pure_action_no_dialogue(self):
        text = "*nods silently*"
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "nods silently")

    def test_pure_dialogue(self):
        text = "Hello, OppaAI!"
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "Hello, OppaAI!")

    def test_display_keeps_markdown_while_removing_structural_metadata(self):
        # No forced ACTION channel: a literal ACTION line is ordinary text
        # now; only EMOTION lines are structural.
        text = (
            "EMOTION: happy\n"
            "ACTION: *waves*\n"
            "Hello, *friend*! Read [the **guide**](https://example.com).\n"
            "- Keep this list item."
        )
        result = format_for_display(text)
        self.assertEqual(
            result,
            "ACTION: *waves*\n"
            "Hello, *friend*! Read [the **guide**](https://example.com).\n"
            "- Keep this list item.",
        )

    def test_no_action_channel_in_parse(self):
        result = parse_aiko_response("EMOTION: happy\nACTION: tilt head\nHi there.")
        self.assertNotIn("action", result)
        self.assertEqual(result["emotion"], "happy")

    def test_natural_action_prose_stays_speakable(self):
        text = "*tilts head* That should work."
        self.assertEqual(extract_dialogue_for_tts(text), "tilts head That should work.")


if __name__ == "__main__":
    unittest.main()
