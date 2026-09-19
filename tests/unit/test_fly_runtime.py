import hashlib
import json
import threading

import pytest

from cognition.fly_runtime import ActiveDynamics, ConnectomeCatalog, Edge, Node, get_fly_runtime
from cognition.fly_runtime.adapters import AvatarMotorController, SensoryObservation
from cognition.fly_runtime.autonomy import ActionScheduler
from cognition.fly_runtime import service
from cognition.neural_state import clear_neural_state, get_neural_state, peek_neural_state


def _catalog():
    return ConnectomeCatalog(
        [Node("eye", "sensory", "optic"), Node("cx", "CX", "central"), Node("dn", "DN", "output")],
        [Edge("eye", "cx", 1.0), Edge("cx", "dn", 1.0)], source="test", version="1",
    )


def test_catalog_path_query_is_bounded_and_source_attributed(tmp_path):
    path = tmp_path / "catalog.json"
    payload = {"source": "fixture", "version": "v1", "nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
    checksum_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(checksum_payload).hexdigest()
    path.write_text(json.dumps({**payload, "checksum": checksum}, indent=2))
    catalog = ConnectomeCatalog.from_path(path)

    result = catalog.subgraph(["a"], budget=1)

    assert catalog.summary()["nodes"] == 2
    assert catalog.summary()["checksum"] == checksum
    assert [node.id for node in result["nodes"]] == ["a"]
    assert result["truncated"] is True


def test_catalog_rejects_a_mismatched_declared_checksum(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"nodes": [], "edges": [], "checksum": "incorrect"}))

    with pytest.raises(ValueError, match="catalog checksum does not match"):
        ConnectomeCatalog.from_path(path)


def test_catalog_frontier_deduplicates_candidates_and_reports_omissions():
    catalog = ConnectomeCatalog(
        [Node("root"), Node("a"), Node("b")],
        [Edge("root", "a"), Edge("root", "a"), Edge("root", "b")],
    )

    result = catalog.subgraph(["root"], budget=2)

    assert [node.id for node in result["nodes"]] == ["a", "root"]
    assert result["edges"] == [Edge("root", "a"), Edge("root", "a")]
    assert result["truncated"] is True


def test_catalog_does_not_report_truncation_for_duplicate_candidates():
    catalog = ConnectomeCatalog([Node("root"), Node("a")], [Edge("root", "a"), Edge("root", "a")])

    result = catalog.subgraph(["root"], budget=2)

    assert [node.id for node in result["nodes"]] == ["a", "root"]
    assert result["truncated"] is False


def test_runtime_is_identity_scoped_and_publishes_bounded_readouts():
    catalog = _catalog()
    clear_neural_state("runtime-a")
    trace = get_fly_runtime("runtime-a", catalog).activate(["eye"], {"eye": 1.0}, budget=3)

    assert trace["active_nodes"] == 3
    assert trace["outputs"]["dn"] > 0
    assert {intent["kind"] for intent in trace["avatar_intents"]} == {"expression", "pose"}
    assert get_neural_state("runtime-a").motion_salience > 0
    assert get_fly_runtime("runtime-a", catalog) is not get_fly_runtime("runtime-b", catalog)


def test_shadow_mode_records_trace_without_publishing_and_preserves_path_details():
    catalog = _catalog()
    clear_neural_state("shadow-runtime")
    trace = get_fly_runtime("shadow-runtime", catalog).activate(
        ["eye"], {"eye": 1.0}, mode="shadow", observations=[{"modality": "visual", "consented": True}],
    )
    assert trace["mode"] == "shadow"
    assert trace["nodes"][0]["id"] == "cx"
    assert trace["edges"][0]["source"] == "eye"
    assert trace["observations"] == [{"modality": "visual", "consented": True}]
    assert peek_neural_state("shadow-runtime") is None


def test_runtime_serializes_trace_publication_per_instance(monkeypatch):
    runtime = get_fly_runtime("locked-runtime", _catalog())
    first_publishing = threading.Event()
    release_first = threading.Event()
    second_published = threading.Event()
    publish_count = 0

    def publish(_rates, _outputs):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 1:
            first_publishing.set()
            assert release_first.wait(timeout=2)
        else:
            second_published.set()

    monkeypatch.setattr(runtime, "_publish", publish)
    first_result = {}
    second_result = {}
    first = threading.Thread(target=lambda: first_result.update(runtime.activate(["eye"], {"eye": 1.0})))
    second = threading.Thread(target=lambda: second_result.update(runtime.activate(["cx"], {"cx": 1.0})))

    first.start()
    assert first_publishing.wait(timeout=2)
    second.start()
    assert not second_published.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert first_result["seeds"] == ["eye"]
    assert second_result["seeds"] == ["cx"]


def test_runtime_publishes_zero_vigor_without_outputs():
    user_id = "runtime-no-outputs"
    clear_neural_state(user_id)
    runtime = get_fly_runtime(user_id, ConnectomeCatalog([Node("eye", "sensory")], []))

    runtime._publish({"eye": 1.0}, {})

    assert get_neural_state(user_id).action_drive == 0.0


def test_rate_dynamics_respects_inhibitory_edges():
    rates = ActiveDynamics([Node("a"), Node("b")], [Edge("a", "b", 1.0, "inhibitory")], leak=1.0)
    rates.step({"a": 1.0, "b": 1.0})
    assert rates.step({})["b"] == 0.0


def test_observation_consent_avatar_allowlist_and_autonomy_boundaries():
    assert SensoryObservation("camera", 1.0, consented=False).drive("eye") == {}
    assert AvatarMotorController().dispatch({"kind": "network_post"}) is False
    scheduler = ActionScheduler(cooldown_s=60)
    assert scheduler.admit({"kind": "pose"}) is True
    assert scheduler.admit({"kind": "pose"}) is False
    assert scheduler.admit({"kind": "network_post"}) is False
    assert scheduler.admit({"kind": "network_post"}, approved=True) is True
    assert [entry["reason"] for entry in scheduler.audit] == ["allowed", "cooldown", "approval_required", "approved"]


def test_service_is_off_without_an_operator_configured_catalog(monkeypatch):
    monkeypatch.delenv("AIKO_FLY_CATALOG_PATH", raising=False)
    monkeypatch.setenv("AIKO_FLY_RUNTIME_MODE", "live")
    assert service.observe("service-user", SensoryObservation("visual", 1.0), seed_types=("visual",)) is None
