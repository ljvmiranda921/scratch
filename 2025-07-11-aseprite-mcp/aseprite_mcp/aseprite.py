import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


class AsepriteCommand:
    """Helper class for running Aseprite commands. Based on diivi/aseprite-mcp's implementation."""

    def __init__(self):
        self.aseprite_path = os.getenv("ASEPRITE_PATH", None)
        if not self.aseprite_path:
            raise ValueError("Value for ASEPRITE_PATH not found!")

    def run_command(self, args: list[str]) -> tuple[bool, Any]:
        """Run an Aseprite command with proper error handling."""
        try:
            cmd = [self.aseprite_path] + args
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    def execute_lua_script(
        self, content: str, filename: Optional[Path] = None
    ) -> tuple[bool, Any]:
        """Runs a Lua script."""
        with tempfile.NamedTemporaryFile(suffix=".lua", delete=False, mode="w") as tmp:
            tmp.write(content)
            script_path = tmp.name

        try:
            args = ["--batch"]
            if filename and filename.exists():
                args.append(str(filename))
            args.extend(["--script", script_path])

            logging.info(f"Running command: {' '.join(args)}")
            return self.run_command(args)

        finally:
            # Teardown temporary file
            os.remove(script_path)
