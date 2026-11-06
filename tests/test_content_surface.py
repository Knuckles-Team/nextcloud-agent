"""CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT R002-R004: skills, prompts and ontology cover the surface."""

import asyncio
import json
import re
from pathlib import Path

import pytest
from scion_sdk.testing.ontology_rollup import orphan_classes, parse_triples

PKG = Path(__file__).resolve().parents[1] / "nextcloud_agent"
TTL = PKG / "ontology" / "nextcloud.ttl"
CP = "http://knuckles.team/kg/connector-pack#"
KG = "http://knuckles.team/kg#"
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _served_tools() -> set[str]:
    from scion_sdk.connector import build_server

    _, server = build_server("nextcloud_agent", command_args=["--transport", "stdio"])
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _component_ids(kind: str) -> set[str]:
    text = TTL.read_text()
    ids = set()
    for block in re.split(r"\s\.\n", text):
        if f"a cp:{kind}" in block:
            match = re.search(r'cp:componentId "([^"]+)"', block)
            if match:
                ids.add(match.group(1))
    return ids


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R002")
def test_ontology_rolls_up_to_core_and_records_every_tool():
    assert orphan_classes(TTL) == set()
    triples = parse_triples(TTL.read_text())
    assert not [t for t in triples if t[1].endswith("#imports")]
    assert _component_ids("Tool") == _served_tools()
    skills = {p.parent.name for p in (PKG / "skills").glob("*/SKILL.md")}
    skills -= {"nextcloud-agent-operations", "nextcloud-agent-development"}
    assert _component_ids("Skill") == skills
    prompts = {p.stem for p in (PKG / "prompts").glob("*.json")}
    assert _component_ids("Prompt") == prompts
    locations = [s for s, p, o in triples if p == TYPE and o == KG + "NextcloudDataLocation"]
    assert len(locations) >= 10


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R003")
def test_prompts_are_agent_definitions_over_real_tools_and_skills():
    tools = _served_tools()
    skills = {p.parent.name for p in (PKG / "skills").glob("*/SKILL.md")}
    for path in (PKG / "prompts").glob("*.json"):
        prompt = json.loads(path.read_text())
        body = prompt["instructions"]
        for key in ("core_directive", "data_locations", "guardrails", "outputs", "workflow"):
            assert body.get(key), (path.name, key)
        assert set(prompt["tools"]) <= tools, path.name
        assert set(prompt["skills"]) <= skills, path.name


@pytest.mark.spec("CONN-NEXTCLOUD-AGENT-CONNECTOR-CONTENT-R004")
def test_skills_and_contracts_name_only_served_tools():
    tools = _served_tools()
    for path in (PKG / "skills").glob("nextcloud-*/SKILL.md"):
        named = set(re.findall(r"`(nextcloud_[a-z_]+)[`(]", path.read_text())) - {"nextcloud_agent"}
        assert named <= tools, (path.parent.name, named - tools)
    step_tools = set(re.findall(r":usesTool nc:tool_([a-z_]+)", TTL.read_text()))
    assert {f"nextcloud_{t}" for t in step_tools} <= tools
    publish = (PKG / "skills" / "nextcloud-publish-asset" / "SKILL.md").read_text()
    for tool in ("nextcloud_copy_item", "nextcloud_create_share", "nextcloud_get_share"):
        assert tool in publish
