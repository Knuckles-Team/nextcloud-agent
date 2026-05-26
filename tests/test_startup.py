import pytest


@pytest.mark.concept("OS-5.4")
def test_server_startup():
    """Validates that the server module parser runs successfully."""
    from nextcloud_agent.agent_server import create_agent_parser

    parser = create_agent_parser()
    args = parser.parse_args(["--debug"])
    assert args.debug is True
