# repo-search

Experimental (see `EXPERIMENTAL`): a semantic code search MCP server for Claude Code, plus the configuration that wires it into a repository.

The server itself, with its setup, tools, settings and internals, is documented in [`tools/repo-search/README.md`](tools/repo-search/README.md).

## Layout

| Path | Purpose |
| --- | --- |
| `tools/repo-search/` | The server (`server.py`), indexer/CLI (`repo_search.py`) and requirements |
| `.mcp.json` | Registers the server with Claude Code for this project |
| `.claude/settings.json` | Pre-approves the read-only tools (`search_code`, `grep_code`, `read_file`); `reindex` still prompts |
| `CLAUDE.md` | Tells Claude when to use `search_code` vs. Grep, and to fall back when the server is down. Add this to your own CLAUDE.md |

## Project-specific configuration

`.mcp.json` sets `HF_HUB_OFFLINE=1`, so the server never contacts Hugging Face. The model must already be in the local cache: run the `index` step from the tool README once (online) before starting Claude Code.
