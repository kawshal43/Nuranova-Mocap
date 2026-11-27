"""Call the installed Blender MCP server over stdio (run with its MCP Python environment)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def execute(code: str) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(command=sys.executable, args=["-m", "blmcp"])
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            available = await session.list_tools()
            if "execute_blender_code" not in {tool.name for tool in available.tools}:
                raise RuntimeError("The Blender MCP server does not expose code execution.")
            reply = await session.call_tool("execute_blender_code", {"code": code})
            result = reply.model_dump(mode="json")
            if reply.isError or (
                reply.structuredContent and reply.structuredContent.get("status") == "error"
            ):
                result["isError"] = True
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reply = asyncio.run(execute(args.code.read_text(encoding="utf-8")))
    text = json.dumps(reply, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.output:
        print(json.dumps({"reply_saved": str(args.output), "isError": reply.get("isError", False)}))
    else:
        print(text)
    if reply.get("isError"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
