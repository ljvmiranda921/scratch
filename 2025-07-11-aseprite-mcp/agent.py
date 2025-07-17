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
    """Simulates an interaction between an agent and an MCP server.
    
    model_name (str): The name of the model to use for the agent.
    mcp_server (MCPServerStdio): The MCP server to connect to.
    request (str): The input request for the agent.
    agent_port (int): The port on which the agent will run.
    workflow_name (str): The name of the workflow for tracing.
    system_prompt (str): The system prompt to initialize the agent.
    """
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
    parser.add_argument("--task_name", "-t", type=str, choices=["simple_art", "spritesheet"], default="simple_art", help="Task name to run the agent on.")
    args = parser.parse_args()
    # fmt: on

    # Set-up the tasks
    task_db = {
        "simple_art": "Create a simple art piece using Aseprite.",
        "spritesheet": "Create a spritesheet for a character using Aseprite.",
    }
    task = task_db.get(args.task_name)

    # Configure the server
    if os.getenv("ASEPRITE_PATH") is None:
        raise ValueError(
            "ASEPRITE_PATH environment variable is not set. Please set it " \
            "to the path of your Aseprite executable."
        )

    aseprite_mcp = MCPServerStdio(
        cache_tools_list=False,
        params={
            "command": "uv",
            "args": ["run", "-m", "aseprite_mcp.server"],
            "env": {"ASEPRITE_PATH": os.getenv("ASEPRITE_PATH", None)},
        },
    )

    # Start the agent interaction with the MCP server
    asyncio.run(
        agent_env_interaction(
            args.model_name,
            aseprite_mcp,
            request=task,
            agent_port=args.port,
        )
    )

    pass
