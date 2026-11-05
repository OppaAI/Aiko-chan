import asyncio
import queue
from interface.webui.webui import AikoWeb
from sensory import vision


def test_observations_are_scoped_and_snapshot_is_stable():
    vision.forget('alice'); vision.forget('bob')
    vision.remember('alice', 'A red cup', 'attachment')
    original = vision.VisualInput('What is this?', vision.context('alice'))
    vision.remember('alice', 'A blue cup', 'attachment')
    assert 'blue' in vision.context('alice') and not vision.context('bob')
    with vision.use_snapshot('alice', original):
        assert 'red' in vision.context('alice') and 'blue' not in vision.context('alice')
    vision.forget('alice')


def test_camera_intent_requires_a_camera_related_request():
    assert vision.camera_intent('Please look at me') == 'capture'
    assert vision.camera_intent('Watch me while I work') == 'watch'
    assert vision.camera_intent('Stop watching me') == 'stop'
    assert vision.camera_intent('Look up the weather') is None


def test_vlm_observation_queues_normal_conversation_not_direct_reply(monkeypatch):
    web = AikoWeb.__new__(AikoWeb)
    web._input_q = queue.Queue()
    sent = []
    web._broadcast = lambda payload, **kw: sent.append(payload)
    web._infer_image = lambda *_: 'A person is holding a cup.'
    monkeypatch.setattr('interface.webui.webui.observe_fly_runtime', lambda *a, **kw: None)
    monkeypatch.setattr('interface.webui.webui._note_visual_frame_bounded', lambda *a: None)
    asyncio.run(web._handle_image_input('image', 'What am I holding?', 'alice'))
    item, uid, _ = web._input_q.get_nowait()
    assert uid == 'alice' and isinstance(item, vision.VisualInput)
    assert 'holding a cup' in item.visual_context
    assert not any(event.get('type') == 'chat' for event in sent)
    vision.forget('alice')
