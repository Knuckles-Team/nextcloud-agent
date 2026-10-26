"""Keep the connector's shared Pages and publication contracts explicit."""

import json
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _actions_on(workflow):
    # PyYAML's YAML 1.1 reader represents the Actions ``on`` key as True.
    return workflow["on"] if "on" in workflow else workflow[True]


def test_pages_publishes_spec_dashboard_and_supports_manual_refresh():
    workflow = _workflow("pages.yml")
    publisher = workflow["jobs"]["publish-pages"]
    push = _actions_on(workflow)["push"]

    assert publisher["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/pages_pipeline.yml@main"
    )
    assert publisher["with"] == {
        "content_source": "docs",
        "spec_dashboard_enabled": True,
        "spec_dashboard_config": ".github/spec-dashboard.json",
    }
    assert "workflow_dispatch" in _actions_on(workflow)
    assert {
        "docs/**",
        "mkdocs.yml",
        "README.md",
        "specs/**",
        ".github/spec-dashboard.json",
        ".github/workflows/pages.yml",
    } <= set(push["paths"])

    config = json.loads((ROOT / ".github/spec-dashboard.json").read_text())
    assert config == {
        "version": 1,
        "sources": [{"glob": "specs/*/status.json", "kind": "spec"}],
    }


def test_single_pages_publisher_and_release_flags_remain_canonical():
    assert not (ROOT / ".github/workflows/docs.yml").exists()

    jobs = _workflow("pipeline.yml")["jobs"]
    assert jobs["publish-pypi"]["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/python_pipeline.yml@main"
    )
    assert jobs["publish-pypi"]["with"] == {"connector": True}
    assert jobs["publish-docker"]["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/container_pipeline.yml@main"
    )
    assert jobs["publish-docker"]["with"] == {"render_connector": True}
    assert jobs["publish-docker"]["needs"] == "publish-pypi"


def test_development_skill_and_runtime_entrypoints_match_source_tree():
    skill = ROOT / "nextcloud_agent/skills/nextcloud-agent-development/SKILL.md"
    content = skill.read_text()

    assert "name: nextcloud-agent-development" in content
    assert "## Rapid delivery: the 5-minute contract" in content
    for path in (
        "nextcloud_agent/__init__.py",
        "nextcloud_agent/auth.py",
        "nextcloud_agent/mcp_server.py",
        "nextcloud_agent/api_client.py",
        "nextcloud_agent/api/api_client_base.py",
        "tests/test_auth.py",
        "tests/test_mcp_server.py",
        "nextcloud_agent/skills/nextcloud-files/SKILL.md",
        "nextcloud_agent/ontology/nextcloud.ttl",
    ):
        assert (ROOT / path).exists(), path

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["nextcloud-mcp"] == "agent_connector_sdk.cli:serve"
    assert project["entry-points"]["agent_connector_sdk.connector"]["nextcloud-mcp"] == (
        "nextcloud_agent:connector"
    )
