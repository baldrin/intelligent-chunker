"""Client-construction TLS wiring in llm.py."""

import sys

import httpx

import intelligent_chunker.llm as llm


def test_tls_client_default_is_none(monkeypatch):
    monkeypatch.delenv("CHUNKER_CA_BUNDLE", raising=False)
    # Simulate truststore not being installed, whatever the venv has.
    monkeypatch.setitem(sys.modules, "truststore", None)
    assert llm._tls_http_client() is None


def test_tls_client_uses_ca_bundle(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setenv("CHUNKER_CA_BUNDLE", "/corp/ca.pem")
    llm._tls_http_client()
    assert captured["verify"] == "/corp/ca.pem"


def test_make_client_wires_custom_http_client(monkeypatch):
    sentinel = httpx.Client()
    monkeypatch.setattr(llm, "_tls_http_client", lambda: sentinel)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = llm.make_client()
    assert client._client is sentinel


def test_make_client_defaults_without_tls_override(monkeypatch):
    monkeypatch.setattr(llm, "_tls_http_client", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = llm.make_client()
    assert isinstance(client._client, httpx.Client)
