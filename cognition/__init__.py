"""Shared context-fetch pool, scoped to each submitting request.

Four workers overlap top-level context reads on the Jetson. Each submission
copies contextvars so per-user identity follows work into a worker. Nested
submissions execute inline: a worker must not wait for a child queued behind
other workers that are themselves waiting for children in this bounded pool.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
import atexit
import threading


class ContextPool(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker_state = threading.local()

    def submit(self, fn, /, *args, **kwargs):
        context = copy_context()

        def run():
            previous = getattr(self._worker_state, "active", False)
            self._worker_state.active = True
            try:
                return context.run(fn, *args, **kwargs)
            finally:
                self._worker_state.active = previous

        if getattr(self._worker_state, "active", False):
            result = Future()
            try:
                result.set_result(run())
            except BaseException as exc:
                result.set_exception(exc)
            return result
        return super().submit(run)


CONTEXT_POOL = ContextPool(max_workers=4, thread_name_prefix="ctx-fetch")
atexit.register(CONTEXT_POOL.shutdown, wait=False)
