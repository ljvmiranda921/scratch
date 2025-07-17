import asyncio

import typer

from agents import Agent, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio
from agents.extensions.models.litellm_model import LitellmModel


def main(
    # fmt: off
    model = typer.Option("--model", "-m", default="gpt-3.5-turbo", help="Model to use for the agent"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the MCP server on"),
    # fmt: on
):
    pass


typer.run(main)