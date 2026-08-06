"""Port helpers for tests.

Tests used to bind well-known fixed ports (8080, 8765, 4249, ...). Those collide with
whatever is already running on the dev machine -- a real observer or robot service, an
unrelated web server on 8080, or a second copy of the suite -- and the failure looks like
a mysterious hang or "address already in use" rather than a test bug. Ask the OS for an
unused port instead.
"""
import socket


def free_port():
    """A port that is unused on 127.0.0.1 right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def free_ports(n):
    """n distinct free ports. All the probe sockets are held open until every port has been
    chosen, so the OS can't hand back the same port twice."""
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()
