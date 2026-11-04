import threading

import pytest

from cognition.knowledge import watcher as module


@pytest.mark.parametrize('operation', ['nested_edit', 'moved_directory'])
def test_recursive_watcher_delivers_files(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(module, 'user_workspace_root', lambda uid: tmp_path)
    nested = tmp_path / 'library' / 'notes'
    nested.mkdir(parents=True)
    target = nested / 'guide.txt'
    target.write_text('old')
    delivered = threading.Event()
    expected = target if operation == 'nested_edit' else tmp_path / 'library/imported/guide.txt'
    def callback(paths):
        if expected in paths:
            delivered.set()
    watcher = module.KnowledgeFolderWatcher(user_id='owner', debounce_seconds=.03, on_files=callback)
    assert watcher.start()
    try:
        if operation == 'nested_edit':
            target.write_text('updated')
        else:
            incoming = tmp_path / 'incoming'
            incoming.mkdir()
            (incoming / 'guide.txt').write_text('imported')
            incoming.rename(tmp_path / 'library/imported')
        assert delivered.wait(4), 'recursive event was not delivered'
    finally:
        watcher.stop()


def test_queue_overflow_requests_library_rescan(tmp_path):
    event = module._INOTIFY_EVENT.pack(-1, module._IN_Q_OVERFLOW, 0, 0)
    assert module._parse_events(event, {}, overflow_root=tmp_path) == [tmp_path]
    assert module._needs_watch_refresh(event)


def test_watcher_rejects_folder_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'user_workspace_root', lambda uid: tmp_path)
    watcher = module.KnowledgeFolderWatcher(knowledge_dir='../outside')
    with pytest.raises(ValueError, match='workspace'):
        _ = watcher.folder
