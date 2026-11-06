"""CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT R002-R005: ontology, prompts, skills and surface contracts."""

import ast
import asyncio
import json
import re
from pathlib import Path

import pytest
from scion_sdk.testing.ontology_rollup import orphan_classes, parse_triples

PKG = Path(__file__).resolve().parents[1] / "nextcloud_agent"
TTL = PKG / "ontology" / "nextcloud.ttl"
CONTRACTS = PKG / "connector_contracts.py"
PLATFORM_SKILLS = {"nextcloud-agent-operations", "nextcloud-agent-development"}


def _served():
    from scion_sdk.connector import build_server

    _, server = build_server("nextcloud_agent", command_args=["--transport", "stdio"])
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _calls(name: str) -> list[ast.Call]:
    tree = ast.parse(CONTRACTS.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == name
    ]


def _kw(call: ast.Call, key: str):
    for keyword in call.keywords:
        if keyword.arg == key:
            return ast.literal_eval(keyword.value)
    return None


def _skills() -> set[str]:
    return {p.parent.name for p in (PKG / "skills").glob("*/SKILL.md")}


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002")
def test_ontology_rolls_up_to_core_without_imports():
    assert orphan_classes(TTL) == set()
    triples = parse_triples(TTL.read_text())
    assert not [t for t in triples if t[1].endswith("#imports")]
    declared = {s for s, p, o in triples if o.endswith("#Class")}
    assert len(declared) >= 16


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R003")
def test_prompts_are_agent_definitions_over_real_tools_and_skills():
    tools, skills = _served(), _skills()
    locations = {ast.literal_eval(c.args[0]) for c in _calls("DataLocation")}
    roles = {ast.literal_eval(c.args[0]) for c in _calls("BusinessRole")}
    kinds = []
    for path in (PKG / "prompts").glob("*.json"):
        prompt = json.loads(path.read_text())
        agent = prompt["agent"]
        kinds.append(agent["kind"])
        assert prompt["identity"]["role"] and prompt["identity"]["goal"], path.name
        assert set(agent["tools"]) <= tools, path.name
        assert set(agent["skills"]) <= skills, path.name
        assert set(agent["data"]["reads"]) | set(agent["data"]["writes"]) <= locations
        assert agent["guardrails"]["never"] and agent["outputs"], path.name
        if agent["kind"] == "business-role":
            assert agent["business_role"] in roles, path.name
    assert "operator" in kinds and kinds.count("business-role") == len(roles)


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R004")
def test_skills_name_only_served_tools():
    tools = _served()
    for path in (PKG / "skills").glob("nextcloud-*/SKILL.md"):
        named = set(re.findall(r"`(nextcloud_[a-z_]+)[`(]", path.read_text()))
        assert named <= tools, (path.parent.name, named - tools)
    publish = (PKG / "skills" / "nextcloud-publish-asset" / "SKILL.md").read_text()
    for tool in ("nextcloud_copy_item", "nextcloud_create_share", "nextcloud_get_share"):
        assert tool in publish


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R005")
def test_contracts_cover_every_tool_prompt_and_skill():
    tools = _served()
    stepped = {ast.literal_eval(c.args[0]) for c in _calls("ToolStep")}
    assert stepped == tools
    assert all(_kw(c, "result") for c in _calls("ToolStep"))
    assert _calls("Refusal")
    links = _calls("SurfaceLink")
    linked_prompts = {p for c in links for p in _kw(c, "prompts")}
    linked_skills = {s for c in links for s in _kw(c, "skills")}
    assert linked_prompts == {p.stem for p in (PKG / "prompts").glob("*.json")}
    assert linked_skills == _skills()
    for call in _calls("DataLocation"):
        assert _kw(call, "read_tool") in tools


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R005")
def test_sdk_surface_check_is_clean_when_available():
    surface = pytest.importorskip("scion_sdk.testing.surface_check")
    assert surface.surface_check("nextcloud_agent") == []
