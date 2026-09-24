from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import ModuleType
import sys


class _Memorize:
    def __init__(self) -> None:
        self.user_id = "github_alice"

    def get_user_id(self) -> str:
        return self.user_id

    def get_display_name(self) -> str:
        return "Alice"

    def get_between(self, *_args, **_kwargs):
        return []


def test_daily_reflection_rechecks_existence_after_lock(monkeypatch, tmp_path):
    """A second runner sees the post produced by the first after locking."""
    import system.schedule as schedule

    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    exists = {"value": False}
    posted = []
    monkeypatch.setattr(schedule, "_reflection_post_exists", lambda _date: exists["value"])
    reflect = ModuleType("cognition.consolidate.reflect")
    reflect.REFLECT_MAX_MEMS = 10
    reflect.filter_reflect_snippets = lambda memories, _date: memories
    monkeypatch.setitem(sys.modules, "cognition.consolidate.reflect", reflect)

    def post(*_args, **_kwargs):
        posted.append(True)
        exists["value"] = True
        return {"success": True}

    runner = schedule.ScheduleRunner(
        memorize=_Memorize(),
        generate_and_post_fn=post,
        user_id="github_alice",
    )
    target = datetime(2026, 8, 25, tzinfo=timezone.utc)
    runner._run_daily_reflect_and_dream(for_date=target)
    runner._run_daily_reflect_and_dream(for_date=target)

    assert posted == [True]


def test_scheduler_lock_is_released_after_critical_section(monkeypatch, tmp_path):
    import system.schedule as schedule

    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    with schedule._scheduler_run_lock("github_alice", "jobs") as acquired:
        assert acquired
        with schedule._scheduler_run_lock("github_alice", "jobs") as contender:
            assert not contender
    with schedule._scheduler_run_lock("github_alice", "jobs") as acquired_after_release:
        assert acquired_after_release


def test_restore_schedule_record_replaces_cancelled_record(monkeypatch):
    import system.schedule as schedule

    jobs = [{"id": "job-1", "enabled": False, "title": "Changed"}]
    written = []
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: jobs)
    monkeypatch.setattr(schedule, "_write_all", lambda records, user_id=None: written.append(records.copy()))

    restored = {"id": "job-1", "enabled": True, "title": "Original"}
    assert schedule.restore_schedule_record(restored)
    assert jobs == [restored]
    assert written == [[restored]]


def test_catchup_retains_date_after_busy_timeout(monkeypatch):
    import system.schedule as schedule

    monkeypatch.setattr(schedule.ScheduleRunner, "_missing_reflection_dates", lambda _self: [])
    monkeypatch.setattr(schedule.ScheduleRunner, "_monthly_catchup_needed", lambda _self: False)
    runner = schedule.ScheduleRunner(user_id="github_alice")
    date = datetime(2026, 8, 25, tzinfo=timezone.utc)
    runner._catchup_dates = [date]
    monkeypatch.setattr(schedule, "acquire_busy", lambda **_kwargs: False)

    runner._run_catchup_backfill()

    assert runner._catchup_dates == [date]


def test_catchup_releases_gate_if_dates_change_while_waiting(monkeypatch):
    import system.schedule as schedule

    monkeypatch.setattr(schedule.ScheduleRunner, "_missing_reflection_dates", lambda _self: [])
    monkeypatch.setattr(schedule.ScheduleRunner, "_monthly_catchup_needed", lambda _self: False)
    runner = schedule.ScheduleRunner(user_id="github_alice")
    runner._catchup_dates = [datetime(2026, 8, 25, tzinfo=timezone.utc)]
    released = []

    def acquire(**_kwargs):
        runner._catchup_dates.clear()
        return True

    monkeypatch.setattr(schedule, "acquire_busy", acquire)
    monkeypatch.setattr(schedule, "release_busy", lambda: released.append(True))

    runner._run_catchup_backfill()

    assert released == [True]


def test_new_owner_email_poll_uses_configured_interval(monkeypatch, tmp_path):
    import system.schedule as schedule

    now = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    graph_path = tmp_path / "schedule_graphs.json"
    graph_path.touch()
    written = []
    monkeypatch.setattr(schedule, "OWNER_EMAIL_POLL_S", 73)
    monkeypatch.setattr(schedule.bioclock, "local_now", lambda: now)
    monkeypatch.setattr(schedule, "schedule_graphs_path", lambda user_id=None: graph_path)
    monkeypatch.setattr(schedule, "_read_schedule_graphs", lambda user_id=None: [])
    monkeypatch.setattr(schedule, "_write_schedule_graphs", lambda graphs, user_id=None: written.extend(graphs))
    monkeypatch.setattr(schedule, "_job_post_social_config", lambda user_id=None: {})

    schedule.ensure_schedule_graphs(user_id="github_alice")

    email = next(graph for graph in written if graph["id"] == "owner_email_poll")
    assert email["trigger"]["interval_seconds"] == 73
    assert datetime.fromisoformat(email["next_due"]) == now + timedelta(seconds=73)


def test_ledger_prunes_every_persistent_user_on_each_start(monkeypatch):
    import system.schedule as schedule

    now = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    pruned = []
    ledger_module = ModuleType("cognition.conscience.ledger")
    ledger_module.ledger_for = lambda user_id: type("Ledger", (), {
        "prune": lambda self: pruned.append(user_id) or 0,
    })()
    monkeypatch.setitem(sys.modules, "cognition.conscience.ledger", ledger_module)
    monkeypatch.setattr(schedule.bioclock, "local_now", lambda: now)
    monkeypatch.setattr(schedule, "all_user_ids", lambda: ["github_alice", "github_bob"])
    monkeypatch.setattr(schedule, "acquire_busy", lambda **kwargs: True)
    monkeypatch.setattr(schedule, "release_busy", lambda: None)
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: [])
    monkeypatch.setattr(schedule, "_read_schedule_graphs", lambda user_id=None: [])
    monkeypatch.setattr(schedule.ScheduleRunner, "_due_user_jobs", lambda self, user_id: False)
    monkeypatch.setattr(schedule.ScheduleRunner, "_due_schedule_graphs", lambda self, user_id: False)
    monkeypatch.setattr(schedule.ScheduleRunner, "_missing_reflection_dates", lambda self: [])
    monkeypatch.setattr(schedule.ScheduleRunner, "_monthly_catchup_needed", lambda self: False)
    monkeypatch.setattr(schedule, "_next_daily_reflect_and_dream", lambda: now + timedelta(days=10))
    monkeypatch.setattr(schedule, "_next_monthly_consolidate", lambda: now + timedelta(days=30))

    for _ in range(2):
        runner = schedule.ScheduleRunner(user_id="github_alice")
        assert runner._next_ledger_prune == now
        sleeps = []

        def wait_seconds(wakeup, seconds):
            sleeps.append(seconds)
            runner.stop()

        monkeypatch.setattr(schedule.bioclock, "wait_seconds", wait_seconds)
        runner._run()
        assert runner._next_ledger_prune == now + timedelta(days=7)
        assert sleeps == [7 * 86400]

    assert pruned == ["github_alice", "github_bob"] * 2
