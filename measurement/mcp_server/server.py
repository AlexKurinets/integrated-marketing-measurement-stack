"""MCP server over published results.

The tools are read-only views plus two scenario tools, all backed by the same MeasurementService as the HTTP
API. An assistant connected to this server answers questions with numbers that come from the model tables,
and the explain tool returns the same plain-English sentences the weekly readout uses.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - older SDK layout
    from mcp.server.fastmcp.exceptions import ToolError

from measurement.service import MeasurementService


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (KeyError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def create_server(service: MeasurementService) -> MCPServer:
    mcp = MCPServer("integrated-marketing-measurement", version="0.4.0")

    @mcp.tool()
    def list_model_runs() -> list[dict]:
        """List published weekly refreshes, newest first, with fit diagnostics."""
        return service.runs()

    @mcp.tool()
    def get_channel_results(run_id: str | None = None) -> list[dict]:
        """MMM output per channel: spend, incremental conversions with a 90% interval, ROI and
        response-curve parameters. Defaults to the latest run."""
        return _guard(service.channel_results, run_id)

    @mcp.tool()
    def get_reconciliation(run_id: str | None = None) -> list[dict]:
        """Harmonized table per channel: MMM, geo test and attribution figures side by side with the harmonized
        figure and which source set it. Defaults to the latest run."""
        return _guard(service.reconciliation, run_id)

    @mcp.tool()
    def explain_channel(channel: str | None = None, run_id: str | None = None) -> list[dict]:
        """Plain-English explanation of how each channel's harmonized figure was reached and how the three
        methods compare. Pass a channel to narrow it."""
        return _guard(service.explain, run_id, channel)

    @mcp.tool()
    def list_geo_tests(status: str | None = None) -> list[dict]:
        """Geo matched-market tests with their design and, when complete, the readout."""
        return service.geo_tests(status)

    @mcp.tool()
    def get_geo_test(test_id: str) -> dict:
        """One geo test: treatment and control markets, dates, lift, interval, p-value and cost per
        incremental conversion."""
        return _guard(service.geo_test, test_id)

    @mcp.tool()
    def get_tv_response(dimension: str | None = None, run_id: str | None = None) -> list[dict]:
        """Spot-level TV response rolled up by network, daypart or creative: airings, cost, attributed sessions and
        cost per response."""
        return _guard(service.tv_rollup, run_id, dimension)

    @mcp.tool()
    def simulate_budget_shift(
        from_channel: str, to_channel: str, weekly_amount: float, run_id: str | None = None
    ) -> dict:
        """What-if: move a weekly amount of spend from one channel to another and report expected weekly incremental
        conversions before and after, using the fitted response curves."""
        return _guard(service.simulate_shift, from_channel, to_channel, weekly_amount, run_id)

    @mcp.tool()
    def propose_reallocation(max_shift_pct: float = 0.2, run_id: str | None = None) -> dict:
        """Greedy reallocation across channels on the fitted response curves, moving no channel by more than
        max_shift_pct of its current weekly budget."""
        return _guard(service.propose_reallocation, max_shift_pct, run_id)

    return mcp


def main() -> None:
    from measurement.repository import get_repository

    service = MeasurementService(get_repository())
    server = create_server(service)
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=int(os.environ.get("MCP_PORT", "8001")),
            stateless_http=True,
        )


if __name__ == "__main__":
    main()
