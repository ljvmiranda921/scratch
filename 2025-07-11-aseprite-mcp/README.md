# Aseprite MCP

This repository is an aseprite-mcp implementation based on [divii/aseprite-mcp](https://github.com/diivi/aseprite-mcp) and as basis for my blog post, ["Draw me a swordsman."](https://ljvmiranda921.github.io/notebook/2025/07/30/draw-me-a-swordsman/)

## Setup and Installation

The installation process assumes you have `uv`.
To get started, run:

```sh
uv sync
source .venv/bin/activate
```

You also need to have the Aseprite executable somewhere in your path (`ASEPRITE_PATH`).
I usually download mine from Steam, so in MacOS, it's usually located at:

```sh
export ASEPRITE_PATH="/Users/<USERNAME>/Library/Application Support/Steam/steamapps/common/Aseprite/Aseprite.app/Contents/MacOS/aseprite"
```

## Running the agent

You can run the agent using the below.

```sh
python agent.py --model_name openai/gpt-4o --task_name simple_art
```

If you are using OpenAI models, be sure to prefix the model name with `openai/`.
Under the hood, I'm using LiteLLM for routing and vLLM (with OpenAI endpoints) for inference.

For vLLM models, it is necessary to figure out the model's approporiate tool call parser.
In addition, I also set the `--enable-auto-tool-choice` to ensure that the model will always resort to using tools.
For example, here's the vLLM command for serving Qwen3:

```sh
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

This command should provide you with a URL and port you can access to (e.g., `http://my.vllm.url:8000`).
To start the agent-env interaction, run:

```sh
python agent.py \
    --model_name Qwen/Qwen3-32B \
    --task_name simple_art \
    --agent_url http://my.vllm.url:8000/v1
```
