"""Shared test fixtures."""

import pytest


@pytest.fixture
def headless_env():
    """A Swift environment with no browser, for stepping scenes in tests."""
    swift = pytest.importorskip("swift")
    import workcell  # noqa: F401  (applies the Swift compatibility patch)

    env = swift.Swift()
    env.launch(headless=True)
    env._send_socket = lambda *args, **kwargs: "0"
    return env
