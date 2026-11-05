"""User-consented camera actions; never access a server-side webcam."""
from agentic.registry import tool


@tool('camera_control', description='Control the current user browser camera. capture/watch require an explicit user camera request; stop ends watching. No image is returned until the browser supplies a frame.',
      props={'action': {'type':'string', 'enum':['capture','watch','stop']}}, required=['action'],
      domain='vision', always_on=True, react=True)
def camera_control(action: str) -> str:
    from interface.webui import auth
    if auth.aiko_web_instance is None:
        return "A connected WebUI browser is required for camera control."
    from sensory.vision import camera_action
    return camera_action(action)
