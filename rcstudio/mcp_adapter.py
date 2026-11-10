"""Optional MCP v2 stdio facade over the already-running local API.

Read-only by default. Pass --allow-write explicitly to permit experiment
creation/stopping. Do not print to stdout: the MCP protocol owns it.
"""
from .client import ensure_server

def start_mcp(allow_write=False):
    try:
        from mcp.server import MCPServer  # MCP Python SDK 2.x
    except ImportError as exc:
        raise RuntimeError(
            "MCP SDK missing. Install: pip install -e '.[mcp]'"
        ) from exc
    client = ensure_server()
    if not allow_write:
        from .client import Client
        client = Client(port=client.port, permission="read")

    server = MCPServer("RCStudio Local Research Tools")

    @server.tool()
    def get_workspace() -> dict:
        """Read current datasets and experiments from local RCStudio."""
        return client.workspace_state()

    @server.tool()
    def list_experiments(limit: int = 20) -> dict:
        """List recent local experiments and their progress."""
        return {"runs": client.runs(max(1, min(limit, 100)))}

    @server.tool()
    def get_experiment(run_id: str) -> dict:
        """Read configuration, status and metrics of an experiment."""
        return client.run(run_id)

    @server.tool()
    def get_reservoir_states(run_id: str, limit: int = 150,
                             units: int = 12) -> dict:
        """Read downsampled neuron state matrix and prediction traces."""
        return client.traces(run_id, limit=min(1000, limit), units=min(48, units))

    @server.tool()
    def get_diagnostics() -> dict:
        """Inspect recent interrupted or failed jobs."""
        return client.diagnostics()

    if allow_write:
        @server.tool()
        def create_experiment(config: dict) -> dict:
            """Schedule one ESN experiment with validated local configuration."""
            return client.submit(config)

        @server.tool()
        def stop_experiment(run_id: str) -> dict:
            """Request a queued or running local experiment to stop."""
            return client.stop(run_id)

    server.run(transport="stdio")
