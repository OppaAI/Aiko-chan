import pytest

from cognition.chat_budget import ContextBudgetError, estimate_messages, fit_chat_request


def test_old_turns_are_removed_without_mutating_history_or_current_task():
    messages = [{'role': 'system', 'content': 'Keep instructions'},
                {'role': 'user', 'content': 'old ' * 1200},
                {'role': 'assistant', 'content': 'old answer'},
                {'role': 'user', 'content': 'Current request'}]
    fitted, reserve = fit_chat_request(messages, 1024, 128)
    assert [m['content'] for m in fitted] == ['Keep instructions', 'Current request']
    assert len(messages) == 4
    assert reserve == 128


def test_optional_recall_can_be_omitted_but_required_instructions_stay():
    messages = [{'role': 'system', 'content': 'REQUIRED\n<memory_context>' + 'fact ' * 1200 + '</memory_context>'},
                {'role': 'user', 'content': 'Current request'}]
    fitted, reserve = fit_chat_request(messages, 1024, 128)
    assert 'REQUIRED' in fitted[0]['content']
    assert 'Omitted for context size' in fitted[0]['content']
    assert fitted[-1]['content'] == 'Current request'
    assert estimate_messages(fitted) + reserve <= 1024 * 0.9
    assert 'fact ' in messages[0]['content']


def test_impossible_request_is_rejected_without_truncating_it():
    with pytest.raises(ContextBudgetError):
        fit_chat_request([{'role': 'system', 'content': 'REQUIRED'},
                          {'role': 'user', 'content': 'request ' * 5000}], 1024, 128)


def test_unicode_counts_more_conservatively_than_ascii():
    assert estimate_messages([{'content': '日' * 100}]) > estimate_messages([{'content': 'a' * 100}])


def test_answer_reserve_can_shrink_for_required_context():
    messages = [{'role': 'system', 'content': 'instruction ' * 100}, {'role': 'user', 'content': 'hi'}]
    fitted, reserve = fit_chat_request(messages, 1024, 700)
    assert fitted == messages
    assert 128 <= reserve < 700
