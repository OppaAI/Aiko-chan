from cognition.memory.redaction import redact_data, redact_text


def test_redaction_handles_common_credentials_without_hiding_contact_facts():
    source = ('api_key=sk-12345678901234567890 password: hunter2 '
              'Authorization: Bearer abcdefghijklmnop user email is aiko@example.org')
    cleaned = redact_text(source)
    assert 'sk-12345678901234567890' not in cleaned
    assert 'hunter2' not in cleaned
    assert 'abcdefghijklmnop' not in cleaned
    assert 'aiko@example.org' in cleaned
    assert redact_text(cleaned) == cleaned
    assert 'secret-value' not in redact_text('AWS_SECRET_ACCESS_KEY=secret-value')


def test_redaction_scrubs_nested_secret_fields_and_json_text():
    result = redact_data({'headers': {'Authorization': 'Bearer abcdefghijklmnop'},
                          'note': '{"client_secret":"topsecret"}',
                          'email': 'owner@example.org'})
    assert result['headers']['Authorization'] == '[redacted]'
    assert 'topsecret' not in result['note']
    assert result['email'] == 'owner@example.org'
    assert redact_text(result['note']) == result['note']
