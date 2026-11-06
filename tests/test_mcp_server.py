import sys
from unittest.mock import patch

import pytest

@pytest.mark.concept("ECO-4.0")
def test_requests_dependency_warning_import_error():
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if "RequestsDependencyWarning" in name or "requests.exceptions" in name:
            raise ImportError("mocked import error")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from scion_sdk.connector import build_server

        _, server = build_server("nextcloud_agent", command_args=["--transport", "stdio"])
        assert server is not None
