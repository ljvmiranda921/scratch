import argparse
import asyncio
import logging

from agents import Agent, Runner, trace
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServer, MCPServerStdio


async def agent_env_interaction(
    model_name: str, mcp_server: MCPServer, *, instruction="Draw me a swordsman", agent_port: int = 8000,
):
    async with mcp_server as server:
        with trace(workflow_name="aseprite_agent"):
            model = None
            agent = Agent(
                name="Assistant",
                instructions="Answer questions about the papers on Semantic Scholar.",
                model=model,
                mcp_servers=[mcp_server],
            )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Simulate interaction between agent and MCP server.")
    parser.add_argument("--model_name", "-n", type=str, default="gpt-4o-mini", help="Name of the model to use.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="vLLM server port of the agent.")
    args = parser.parse_args()
    # fmt: on

    aseprite_mcp = MCPServerStdio(
        cache_tools_list=False, params={"command": "uv", "args": "", "env": ""}
    )

    asyncio.run(agent_env_interaction(args.model_name, aseprite_mcp, agent_port=args.port,))

    pass
