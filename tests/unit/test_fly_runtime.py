import json

from cognition.fly_runtime import ActiveDynamics, ConnectomeCatalog, Edge, Node, get_fly_runtime
from cognition.fly_runtime.adapters import AvatarMotorController, SensoryObservation
from cognition.fly_runtime.autonomy import ActionScheduler
from cognition.neural_state import clear_neural_state, get_neural_state


def _catalog():
    return ConnectomeCatalog(
        [Node("eye", "sensory", "optic"), Node("cx", "CX", "central"), Node("dn", "DN", "output")],
        [Edge("eye", "cx", 1.0), Edge("cx", "dn", 1.0)], source="test", version="1",
    )


def test_catalog_path_query_is_bounded_and_source_attributed(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"source": "fixture", "version": "v1", "nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}))
    catalog = ConnectomeCatalog.from_path(path)

    result = catalog.subgraph(["a"], budget=1)

    assert catalog.summary()["nodes"] == 2
    assert catalog.summary()["checksum"]
    assert [node.id for node in result["nodes"]] == ["a"]
    assert result["truncated"] is True


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
    assert get_neural_state("shadow-runtime").motion_salience == 0.0


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
