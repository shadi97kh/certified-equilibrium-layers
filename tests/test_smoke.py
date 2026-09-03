"""Smoke test: package imports and version is set."""

import cel


def test_package_imports() -> None:
    assert hasattr(cel, "__version__")
    assert isinstance(cel.__version__, str)
