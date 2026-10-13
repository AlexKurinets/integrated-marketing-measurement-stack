import asyncio
import json

import pytest

from measurement.mcp_server.server import create_server
from measurement.service import MeasurementService

EXPECTED_TOOLS = {
    "list_model_runs",
    "get_channel_results",
    "get_reconciliation",
    "explain_channel",
    "list_geo_tests",
    "get_geo_test",
    "get_tv_response",
    "simulate_budget_shift",
    "propose_reallocation",
}


@pytest.fixture(scope="module")
def server(fitted_repo):
    return create_server(MeasurementService(fitted_repo, value=180.0))


def _call(server, name, **arguments):
    return asyncio.run(server.call_tool(name, arguments))


def _payload(result):
    sc = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if sc:
        return sc["result"] if isinstance(sc, dict) and "result" in sc else sc
    items = [json.loads(c.text) for c in result.content]
    return items[0] if len(items) == 1 else items


def _is_error(result):
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


def test_tools_are_registered(server):
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description for t in tools)


def test_reconciliation_tool_returns_the_table(server):
    res = _call(server, "get_reconciliation")
    assert not _is_error(res)
    rows = _payload(res)
    assert len(rows) == 5
    assert {"channel", "harmonized_incremental", "harmonized_source", "attributed_conversions"} <= set(rows[0])


def test_explain_and_geo_test_tools(server):
    text = _payload(_call(server, "explain_channel", channel="ctv"))
    assert len(text) == 1 and text[0]["text"].startswith("CTV:")
    t = _payload(_call(server, "get_geo_test", test_id="gt-ctv-spend-up-2026q2"))
    assert t["channel"] == "ctv" and t["readout"]["test_weeks"] == 6


def test_scenario_tool_and_error_handling(server):
    sim = _payload(
        _call(server, "simulate_budget_shift", from_channel="linear_tv", to_channel="ctv", weekly_amount=300.0)
    )
    assert "delta_pct" in sim and sim["run_id"]
    try:
        err = _call(server, "get_channel_results", run_id="does-not-exist")
    except Exception as exc:  # the SDK may raise the tool error directly when called in-process
        assert "does-not-exist" in str(exc) or "Unknown run" in str(exc)
    else:
        assert _is_error(err)
