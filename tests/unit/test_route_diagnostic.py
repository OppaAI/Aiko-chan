"""Standalone route CLI delegates to the live policy without loading it on import."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


def diagnostic():
    path = Path(__file__).parents[1] / 'integration/test_route.py'
    spec = importlib.util.spec_from_file_location('route_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_uses_canonical_router_and_reports_mismatches(capsys):
    module = diagnostic()
    router = Mock()
    router._route_intent.side_effect = [('greeting', None), ('agentic', None)]
    assert module.run_suite(router, [('hello', 'greeting'), ('help me', 'chat')], quiet=True) == 1
    assert router._route_intent.call_count == 2
    assert '1/2' in capsys.readouterr().out


def test_invalid_file_fails_before_starting_models(tmp_path, monkeypatch):
    module = diagnostic()
    path = tmp_path / 'prompts.json'
    path.write_text('[{"prompt":"Hello", "expected":"invalid"}]')
    factory = Mock()
    monkeypatch.setattr(module, '_make_router', factory)
    with pytest.raises(SystemExit):
        module.main(['--file', str(path)])
    factory.assert_not_called()
