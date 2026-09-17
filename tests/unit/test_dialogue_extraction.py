"""
test_dialogue_extraction.py
Tests for extracting spoken dialogue from Aiko messages containing emoji headers and non-verbal cues.
"""

import unittest
from sensory.speak import extract_dialogue_for_tts, format_for_display


class TestDialogueExtraction(unittest.TestCase):

    def test_emoji_header_and_non_verbal(self):
        text = "😊: *sighs softly* (inner thoughts: glad he asked) I'm doing well, thank you."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "I'm doing well, thank you.")

    def test_angry_emoji_and_actions(self):
        text = "😒: *crosses arms* You really don't know? *looks away* Let me explain."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "You really don't know? Let me explain.")

    def test_bracketed_feelings(self):
        text = "🤖: [analyzing parameters...] Here are the results."
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "Here are the results.")

    def test_pure_action_no_dialogue(self):
        text = "*nods silently*"
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "")

    def test_pure_dialogue(self):
        text = "Hello, OppaAI!"
        result = extract_dialogue_for_tts(text)
        self.assertEqual(result, "Hello, OppaAI!")

    def test_display_keeps_markdown_while_removing_structural_metadata(self):
        text = (
            "EMOTION: happy\n"
            "ACTION: *waves*\n"
            "Hello, *friend*! Read [the **guide**](https://example.com).\n"
            "- Keep this list item."
        )
        result = format_for_display(text)
        self.assertEqual(
            result,
            "Hello, *friend*! Read [the **guide**](https://example.com).\n"
            "- Keep this list item.",
        )


if __name__ == "__main__":
    unittest.main()
