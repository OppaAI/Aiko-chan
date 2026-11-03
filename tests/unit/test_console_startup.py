from io import StringIO
import threading

from interface.console import StartupConsole


class Terminal(StringIO):
    def isatty(self):
        return True


def test_redirected_output_is_readable_and_has_no_terminal_controls():
    stream = StringIO()
    display = StartupConsole({"memory": "Load memory"}, stream=stream)
    display.loading("memory")
    display.done("memory")
    display.finish("Ready for input")
    output = stream.getvalue()
    assert "Load memory" in output and "OK" in output and "READY" in output
    assert "\r" not in output and "\033" not in output


def test_parallel_completion_keeps_other_steps_active(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("LOG_CONSOLE", raising=False)
    stream = Terminal()
    display = StartupConsole({"memory": "Load memory", "think": "Start cognition"}, stream=stream)
    display.loading("memory")
    display.loading("think")
    display.done("memory")
    assert "Start cognition" in stream.getvalue().split("\033[2K")[-1]
    display.finish()
    output = stream.getvalue()
    display.done("think")
    assert stream.getvalue() == output


def test_terminal_spinner_advances_and_stops(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("LOG_CONSOLE", raising=False)
    frame_seen = threading.Event()
    class ObservedTerminal(Terminal):
        def write(self, text):
            if text.startswith("  /"):
                frame_seen.set()
            return super().write(text)
    stream = ObservedTerminal()
    display = StartupConsole({"memory": "Load memory"}, stream=stream)
    display.loading("memory")
    stop = threading.Event()
    worker = threading.Thread(target=display.spin, args=(stop,))
    worker.start()
    try:
        assert frame_seen.wait(2)
    finally:
        stop.set()
        worker.join(2)
    assert not worker.is_alive()


def test_no_color_disables_color_but_keeps_progress(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("NO_COLOR", "1")
    stream = Terminal()
    display = StartupConsole({}, stream=stream)
    assert "\033[1;36m" not in stream.getvalue()
    display.loading("Test step")
    display.skip("Test step")
    assert "SKIP" in stream.getvalue()
