import argparse
import asyncio
import logging
import os

from agents import Agent, Runner, trace
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio


async def agent_env_interaction(
    model_name: str,
    mcp_server: MCPServerStdio,
    request: str,
    *,
    agent_port: int = 8000,
    workflow_name: str = "aseprite_agent",
    system_prompt: str = "You are a function-calling agent that can use tools to perform a given task.",
):
    async with mcp_server as server:
        with trace(workflow_name=workflow_name):
            # TODO: Configure the model properly
            model = None
            agent = Agent(
                name="Assistant",
                instructions=system_prompt,
                model=model,
                mcp_servers=[server],
            )

            result = await Runner.run(starting_agent=agent, input=request)
            # TODO: Use rich for chat-like formatting
            print(result.final_output)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Simulate interaction between agent and MCP server.")
    parser.add_argument("--model_name", "-n", type=str, default="gpt-4o-mini", help="Name of the model to use.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="vLLM server port of the agent.")
    args = parser.parse_args()
    # fmt: on

    aseprite_mcp = MCPServerStdio(
        cache_tools_list=False,
        params={
            "command": "uv",
            "args": ["run", "-m", "aseprite_mcp.server"],
            "env": dict(os.environ),
        },
    )

    asyncio.run(
        agent_env_interaction(
            args.model_name,
            aseprite_mcp,
            agent_port=args.port,
        )
    )

    pass
