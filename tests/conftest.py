import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test runs offline with its own empty data dir."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access in tests is not allowed")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("FFCALC_DATA", str(tmp_path / "data"))
