from contextvars import ContextVar

import pytest

from cognition import ContextPool


def test_submissions_propagate_isolated_request_identity():
    identity = ContextVar('test_identity', default='guest')
    with ContextPool(max_workers=1) as pool:
        token = identity.set('alice')
        try:
            first = pool.submit(identity.get)
            identity.set('bob')
            second = pool.submit(identity.get)
            assert first.result(2) == 'alice'
            assert second.result(2) == 'bob'
        finally:
            identity.reset(token)
        assert pool.submit(identity.get).result(2) == 'guest'


def test_nested_submission_does_not_starve_single_worker_pool():
    identity = ContextVar('nested_identity', default='guest')
    with ContextPool(max_workers=1) as pool:
        token = identity.set('alice')
        try:
            future = pool.submit(lambda: pool.submit(identity.get).result(timeout=1))
            assert future.result(2) == 'alice'
        finally:
            identity.reset(token)


def test_nested_errors_propagate_without_poisoning_worker():
    with ContextPool(max_workers=1) as pool:
        def parent():
            child = pool.submit(lambda: 1 / 0)
            with pytest.raises(ZeroDivisionError):
                child.result()
            return pool.submit(lambda: 'recovered').result()
        assert pool.submit(parent).result(2) == 'recovered'
