from ace_next.integrations.mcp import adapters, server


def test_create_server_requires_mcp_extra(monkeypatch):
    def fake_import_module(name: str):
        err = ModuleNotFoundError("No module named 'mcp'")
        err.name = "mcp"
        raise err

    monkeypatch.setattr(server, "import_module", fake_import_module)

    try:
        server.create_server()
    except RuntimeError as exc:
        assert "ace-framework[mcp]" in str(exc)
    else:  # pragma: no cover - defensive check
        raise AssertionError("create_server() should require the mcp extra")


def test_register_tools_requires_mcp_extra(monkeypatch):
    def fake_import_module(name: str):
        err = ModuleNotFoundError("No module named 'mcp'")
        err.name = "mcp"
        raise err

    monkeypatch.setattr(adapters, "import_module", fake_import_module)

    try:
        adapters.register_tools(object(), object())  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "ace-framework[mcp]" in str(exc)
    else:  # pragma: no cover - defensive check
        raise AssertionError("register_tools() should require the mcp extra")
