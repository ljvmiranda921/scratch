import base64
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from aseprite_mcp.aseprite import AsepriteCommand

cmd = AsepriteCommand()
mcp = FastMCP("aseprite-mcp")


@mcp.tool()
async def create_canvas(
    width: int, height: int, filename: Path = Path("canvas.aseprite")
) -> str:
    """Create a new Aseprite canvas with the specific dimensions

    width (int): width of the canvas in pixels.
    height (int): height of the canvas in pixels
    filename (Path): name of the output file.
    """

    script = f"""
    local spr = Sprite({width}, {height})
    spr:saveAs("{filename}")
    return "Canvas created successfully: {filename}"
    """

    success, output = cmd.execute_lua_script(script)
    output_message = (
        f"Canvas created successfully: {filename}"
        if success
        else f"Failed to create canvas: {output}"
    )
    return output_message


@mcp.tool()
async def add_layer(filename: Path, layer_name: str) -> str:
    """Add a new layer to the Aseprite file.

    filename (Path): name of the Aseprite file to modify or add layer to.
    layer_name (str): name of the new layer.
    """
    if not filename.exists():
        return f"File {filename} not found."

    script = f"""
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        spr:newLayer()
        app.activeLayer.name = "{layer_name}"

    spr:saveAs(spr.filename)
    return "Layer added successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)
    output_message = (
        f"Layer '{layer_name}' added successfully to {filename}"
        if success
        else f"Failed to add layer: {output}"
    )
    return output_message


@mcp.tool()
async def add_frame(filename: Path) -> str:
    """Add a new layer to the Aseprite file.

    filename (Path): name of the Aseprite file to modify or add frame to.
    """
    if not filename.exists():
        return f"File {filename} not found"

    script = """
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        spr:newFrame()
    end)

    spr:saveAs(spr.filename)
    return "Frame added successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)
    output_message = (
        f"New frame added successfully to {filename}"
        if success
        else f"Failed to add frame: {output}"
    )
    return output_message


@mcp.tool()
async def draw_pixels(filename: Path, pixels: list[dict[str, Any]]) -> str:
    """Draw pixels on the canvas with specified colors.

    filename (Path): Name of the Aseprite file to modify
    pixels (list[dict[str, Any]]): List of pixel data, each containing: {"x": int, "y": int, "color": str} where color is a hex code like "#FF0000"
    """
    if not filename.exists():
        return f"File {filename} not found"

    script = """
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        local cel = app.activeCel
        if not cel then
            -- If no active cel, create one
            app.activeLayer = spr.layers[1]
            app.activeFrame = spr.frames[1]
            cel = app.activeCel
            if not cel then
                return "No active cel and couldn't create one"
            end
        end

        local img = cel.image
    """

    # Add pixel drawing commands
    for pixel in pixels:
        x, y = pixel.get("x", 0), pixel.get("y", 0)
        color = pixel.get("color", "#000000")
        # Convert hex to RGB
        color = color.lstrip("#")
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

        script += f"""
        img:putPixel({x}, {y}, Color({r}, {g}, {b}, 255))
        """

    script += """
    end)

    spr:saveAs(spr.filename)
    return "Pixels drawn successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)
    output_message = (
        f"Pixels drawn successfuly in {filename}"
        if success
        else f"Failed to draw pixels: {output}"
    )
    return output_message


@mcp.tool()
async def draw_line(
    filename: Path,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "#000000",
    thickness: int = 1,
) -> str:
    """Draw a line on the canvas.

    filename (Path): Name of the Aseprite file to modify
    x1 (int): Starting x coordinate
    y1 (int): Starting y coordinate
    x2 (int): Ending x coordinate
    y2 (int): Ending y coordinate
    color (str): Hex color code (default: "#000000")
    thickness (int): Line thickness in pixels (default: 1)
    """
    if not filename.exists():
        return f"File {filename} not found"

    # Convert hex to RGB
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    script = f"""
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        local cel = app.activeCel
        if not cel then
            app.activeLayer = spr.layers[1]
            app.activeFrame = spr.frames[1]
            cel = app.activeCel
            if not cel then
                return "No active cel and couldn't create one"
            end
        end

        local color = Color({r}, {g}, {b}, 255)
        local brush = Brush()
        brush.size = {thickness}
        app.useTool({{
            tool="line",
            color=color,
            brush=brush,
            points={{Point({x1}, {y1}), Point({x2}, {y2})}}
        }})
    end)

    spr:saveAs(spr.filename)
    return "Line drawn successfully"
    """
    success, output = cmd.execute_lua_script(script, filename)

    output_message = (
        f"Line drawn successfully in {filename}"
        if success
        else f"Failed to draw line: {output}"
    )
    return output_message


@mcp.tool()
async def draw_rectangle(
    filename: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    color: str = "#000000",
    fill: bool = False,
) -> str:
    """Draw a rectangle on the canvas.

    filename (Path): Name of the Aseprite file to modify
    x (int): Top-left x coordinate
    y (int): Top-left y coordinate
    width (int): Width of the rectangle
    height (int): Height of the rectangle
    color (str): Hex color code (default: "#000000")
    fill (bool): Whether to fill the rectangle (default: False)
    """
    if not filename.exists():
        return f"File {filename} not found"

    # Convert hex to RGB
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    script = f"""
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        local cel = app.activeCel
        if not cel then
            app.activeLayer = spr.layers[1]
            app.activeFrame = spr.frames[1]
            cel = app.activeCel
            if not cel then
                return "No active cel and couldn't create one"
            end
        end

        local color = Color({r}, {g}, {b}, 255)
        local tool = {'"rectangle"' if not fill else '"filled_rectangle"'}
        app.useTool({{
            tool=tool,
            color=color,
            points={{Point({x}, {y}), Point({x + width}, {y + height})}}
        }})
    end)

    spr:saveAs(spr.filename)
    return "Rectangle drawn successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)

    output_message = (
        f"Rectangle drawn successfully in {filename}"
        if success
        else f"Failed to draw rectangle: {output}"
    )
    return output_message


@mcp.tool()
async def fill_area(filename: Path, x: int, y: int, color: str = "#000000") -> str:
    """Fill an area with color using the paint bucket tool.

    filename (Path): Name of the Aseprite file to modify
    x (int): X coordinate to fill from
    y (int): Y coordinate to fill from
    color (str): Hex color code (default: "#000000")
    """
    if not filename.exists():
        return f"File {filename} not found"

    # Convert hex to RGB
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    script = f"""
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        local cel = app.activeCel
        if not cel then
            app.activeLayer = spr.layers[1]
            app.activeFrame = spr.frames[1]
            cel = app.activeCel
            if not cel then
                return "No active cel and couldn't create one"
            end
        end

        local color = Color({r}, {g}, {b}, 255)
        app.useTool({{
            tool="paint_bucket",
            color=color,
            points={{Point({x}, {y})}}
        }})
    end)

    spr:saveAs(spr.filename)
    return "Area filled successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)

    output_message = (
        f"Area filled successfully in {filename}"
        if success
        else f"Failed to fill area: {output}"
    )
    return output_message


@mcp.tool()
async def draw_circle(
    filename: Path,
    center_x: int,
    center_y: int,
    radius: int,
    color: str = "#000000",
    fill: bool = False,
) -> str:
    """Draw a circle on the canvas.

    filename (Path): Name of the Aseprite file to modify
    center_x (int): X coordinate of circle center
    center_y (int): Y coordinate of circle center
    radius (int): Radius of the circle in pixels
    color (str): Hex color code (default: "#000000")
    fill (bool): Whether to fill the circle (default: False)
    """
    if not filename.exists():
        return f"File {filename} not found"

    # Convert hex to RGB
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    script = f"""
    local spr = app.activeSprite
    if not spr then return "No active sprite" end

    app.transaction(function()
        local cel = app.activeCel
        if not cel then
            app.activeLayer = spr.layers[1]
            app.activeFrame = spr.frames[1]
            cel = app.activeCel
            if not cel then
                return "No active cel and couldn't create one"
            end
        end

        local color = Color({r}, {g}, {b}, 255)
        local tool = {'"ellipse"' if not fill else '"filled_ellipse"'}
        app.useTool({{
            tool=tool,
            color=color,
            points={{
                Point({center_x - radius}, {center_y - radius}),
                Point({center_x + radius}, {center_y + radius})
            }}
        }})
    end)

    spr:saveAs(spr.filename)
    return "Circle drawn successfully"
    """

    success, output = cmd.execute_lua_script(script, filename)

    output_message = (
        f"Circle drawn successfully in {filename}"
        if success
        else f"Failed to draw circle: {output}"
    )
    return output_message


@mcp.tool()
async def export_sprite(
    filename: Path, output_filename: Path, format: str = "png"
) -> str:
    """Export the Aseprite file to another format.

    filename (Path): Name of the Aseprite file to export
    output_filename (Path): Name of the output file
    format (str): Output format (default: "png", can be "png", "gif", "jpg", etc.)
    """
    if not filename.exists():
        return f"File {filename} not found"

    # Make sure format is lowercase
    format = format.lower()

    # Ensure output filename has the correct extension
    if not output_filename.suffix.lower() == f".{format}":
        output_filename = output_filename.with_suffix(f".{format}")

    # For animated exports
    if format == "gif":
        args = ["--batch", str(filename), "--save-as", str(output_filename)]
        success, output = cmd.run_command(args)
    else:
        # For still image exports
        args = ["--batch", str(filename), "--save-as", str(output_filename)]
        success, output = cmd.run_command(args)

    output_message = (
        f"Sprite exported successfully to {output_filename}"
        if success
        else f"Failed to export sprite: {output}"
    )
    return output_message


@mcp.tool()
def preview_image(filename: Path) -> str:
    """Read an image file and return it as base64 data for display

    filename (Path): path to the PNG file to show.
    RETURNS base64 version of the image to preview.
    """
    try:
        if filename.exists():
            with open(filename, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{image_data}"
        else:
            return f"File not found: {filename}"
    except Exception as e:
        return f"Error reading image: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
