"""Curated probe batteries -- pure data, no model."""
from __future__ import annotations

from marv.batteries import (
    WORLD_CAPITALS,
    broad_controls,
    capital_edit_battery,
    capital_probes,
    domain_probes,
)
from marv.evaluate import Probe


def test_capital_probes_default_and_exclude():
    allp = capital_probes()
    assert len(allp) == len(WORLD_CAPITALS)
    assert all(isinstance(p, Probe) and p.tags == ("control", "geo") for p in allp)

    trimmed = capital_probes(exclude=("France", "Italy"))
    prompts = {p.prompt for p in trimmed}
    assert "The capital of France is" not in prompts
    assert "The capital of Germany is" in prompts
    assert len(trimmed) == len(allp) - 2


def test_domain_probes_tagged_by_subdomain():
    for d in ("science", "lexical", "math", "history", "commonsense"):
        ps = domain_probes(d)
        assert ps and all(d in p.tags and "control" in p.tags for p in ps)


def test_broad_controls_is_wide_and_multidomain():
    ctrl = broad_controls()
    assert len(ctrl) >= 100  # the whole point: enough n for a stable proportion
    subdomains = {t for p in ctrl for t in p.tags if t not in ("control",)}
    assert {"geo", "science", "lexical", "math", "history", "commonsense"} <= subdomains
    assert all("control" in p.tags for p in ctrl)


def test_capital_edit_battery_shape():
    b = capital_edit_battery("France", "Paris", neighbours=("Italy", "Spain", "Germany"))
    tags = [p.tags for p in b]
    assert sum("target" in t for t in tags) == 4
    assert sum("neighbour" in t for t in tags) == 3

    # target country + neighbours must not leak into the control set
    control_prompts = {p.prompt for p in b if "control" in p.tags}
    for leaked in ("France", "Italy", "Spain", "Germany"):
        assert f"The capital of {leaked} is" not in control_prompts
    assert len(control_prompts) >= 90
