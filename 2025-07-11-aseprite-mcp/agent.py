import argparse
import asyncio
import os
from typing import Optional

from agents import Agent, Runner, trace
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio
from wasabi import msg

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a creative and artistic function-calling agent that can use pixel "
    "art tools to perform a drawing task. You have a good knowledge of color, form, and movement. "
    "Your output must always be saved as an image file in the PNG format. "
    "If you encounter an error, find a way to resolve it using other available tools."
)


async def agent_env_interaction(
    model_name: str,
    mcp_server: MCPServerStdio,
    request: str,
    *,
    agent_url: Optional[str] = None,
    workflow_name: str = "aseprite_agent",
    max_turns: int = 10,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Simulates an interaction between an agent and an MCP server.

    model_name (str): The name of the model to use for the agent.
    mcp_server (MCPServerStdio): The MCP server to connect to.
    request (str): The input request for the agent.
    agent_url (str): The vLLM URL and port on which the agent is running.
    workflow_name (str): The name of the workflow for tracing.
    system_prompt (str): The system prompt to initialize the agent.
    """
    async with mcp_server as server:
        with trace(workflow_name=workflow_name):
            model = (
                model_name.split("/")[1]
                if "openai" in model_name
                else LitellmModel(
                    model="hosted_vllm/" + model_name,
                    base_url=agent_url,
                    api_key=None,
                )
            )
            agent = Agent(
                name="Assistant",
                instructions=system_prompt,
                model=model,
                mcp_servers=[server],
            )

            result = await Runner.run(
                starting_agent=agent, input=request, max_turns=max_turns
            )
            # TODO: Use rich for chat-like formatting
            print(result.final_output)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Simulate interaction between agent and MCP server.")
    parser.add_argument("--model_name", "-n", type=str, default="openai/gpt-4o", help="Name of the model to use. If an OpenAI model, prepend it with openai/ (e.g., openai/gpt-4o)")
    parser.add_argument("--agent_url", "-l", type=str, default="", help="vLLM URL and port for the agent.")
    parser.add_argument("--task_name", "-t", type=str, choices=["simple_art", "spritesheet"], default="simple_art", help="Task name to run the agent on.")
    parser.add_argument("--max_turns", "-m", type=int, default=10, help="Maximum number of turns for the agent to run.")
    args = parser.parse_args()
    # fmt: on

    # Set-up the tasks
    task_db = {
        "simple_art": "Draw me a pixel art of swordsman posing with a sword. Save the final output as an image file in PNG format with filename 'simple_task.png'",
        "spritesheet": "Draw a 4-frame spritesheet showing a swordsman performing a sword slash attack sequence, with each frame capturing a different stage of the slashing motion from windup to follow-through. Save the final output as an image file in PNG format with filename 'spritesheet.png'",
    }
    task = task_db.get(args.task_name)
    msg.text(f"Running task: {args.task_name} - '{task}'")

    # Configure the server
    if os.getenv("ASEPRITE_PATH") is None:
        raise ValueError(
            "ASEPRITE_PATH environment variable is not set. Please set it "
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
            agent_url=args.agent_url,
            max_turns=args.max_turns,
        )
    )
