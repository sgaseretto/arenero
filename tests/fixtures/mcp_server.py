"""Small standard-input MCP server used by integration tests."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("arenero-test")


@server.tool(description="Multiply two integers")
def multiply(a: int, b: int) -> dict[str, int]:
    """Multiply two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        A dictionary with the product.
    """
    return {"product": a * b}


if __name__ == "__main__":
    server.run()
