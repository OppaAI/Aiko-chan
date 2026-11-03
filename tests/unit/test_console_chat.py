from io import StringIO

from interface.console import ChatConsole


class Terminal(StringIO):
    def isatty(self):
        return True


def test_interrupted_stream_places_diagnostics_on_separate_lines():
    stream = StringIO()
    ui = ChatConsole(stream)
    ui.begin()
    ui.token('Checking')
    ui.message('sys', '[tools] repo_read_file(example.py)')
    ui.token('Here is the result.')
    ui.commit()
    assert stream.getvalue() == '\n  Aiko > Checking\n  Tools > repo_read_file(example.py)\n\n  Aiko > Here is the result.\n'


def test_plain_output_strips_upstream_ansi_and_indents_debug_lines():
    stream = StringIO()
    ui = ChatConsole(stream)
    ui.message('sys', '\033[35m[mem][+2ms][30 tok] first\nsecond\033[0m')
    assert '\033' not in stream.getvalue()
    assert 'Mem > [+2ms][30 tok] first\n    second\n' in stream.getvalue()


def test_role_colors_are_distinct_and_no_color_is_respected(monkeypatch):
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    stream = Terminal()
    ui = ChatConsole(stream)
    ui.message('you', 'hello')
    ui.message('aiko', 'hi')
    assert '\033[36mYou > ' in stream.getvalue()
    assert '\033[35mAiko > ' in stream.getvalue()
    monkeypatch.setenv('NO_COLOR', '1')
    plain = Terminal()
    ChatConsole(plain).message('sys', '\033[31m[thinking] working\033[0m')
    assert '\033' not in plain.getvalue()


def test_multiline_reply_keeps_one_role_label():
    stream = StringIO()
    ui = ChatConsole(stream)
    ui.begin()
    ui.token('first\n')
    ui.token('second')
    ui.commit()
    assert stream.getvalue() == '\n  Aiko > first\nsecond\n'


def test_cli_control_markers_do_not_leak_into_chat(capsys):
    from interface.cli.cli import AikoSimpleCLI
    ui = AikoSimpleCLI()
    capsys.readouterr()
    ui.turn_start()
    ui.stream_token('__THINKING__')
    ui.stream_token('__TOOL__:repo_read_file(example.py)')
    ui.stream_token('Done.')
    ui.stream_commit()
    text = capsys.readouterr().out
    assert '__THINKING__' not in text and '__TOOL__' not in text
    assert 'Thinking > step 1' in text and 'Tools > repo_read_file' in text
    assert 'Aiko > Done.' in text


def test_cli_does_not_repeat_echoed_typed_input(monkeypatch, capsys):
    from interface.cli.cli import AikoSimpleCLI
    ui = AikoSimpleCLI()
    capsys.readouterr()
    def typed(prompt):
        print(prompt + 'Hello Aiko')
        return 'Hello Aiko'
    monkeypatch.setattr('builtins.input', typed)
    text = ui.get_input()
    ui.add_message('you', text)
    assert capsys.readouterr().out.count('Hello Aiko') == 1


def test_trace_and_metrics_have_readable_plain_labels():
    stream = StringIO()
    ui = ChatConsole(stream)
    ui.message('sys', '🧠  ── turn #1 [text] ──')
    ui.message('sys', '[metrics] 12 tok/s | tokens in/out: 80/20')
    assert 'Trace > ── turn #1' in stream.getvalue()
    assert 'Metrics > 12 tok/s' in stream.getvalue()
