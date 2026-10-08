# repo-search

An MCP server that lets a coding agent answer questions about a Git repository. It embeds the code with EmbeddingGemma 2 (text/code encoder only, 270M parameters) and exposes four tools:

| Tool | Use |
| --- | --- |
| `search_code` | Semantic search: "where do we validate the session cookie?" |
| `grep_code` | Exact regex search (`git grep`) for identifiers, strings, callers |
| `read_file` | Read a line range around a hit |
| `reindex` | Update the index; only changed files are re-embedded |

Before you rely on it, check these on your machine:

- **Prompt names**: the code uses `SearchQuery` and `Document` from the guide. The README has a one-liner that prints the model's full prompt list, in case there is a code-specific one; both names are overridable by environment variable.
- **MCP SDK version**: the server targets mcp 2.x, where `FastMCP` was renamed `MCPServer`. Most examples online still show the 1.x import, which no longer works.
- **Launch path**: the `.mcp.json` uses a relative path to `server.py`, which assumes Claude Code is started at the repo root. I haven't confirmed how it resolves from a subdirectory.
- **Shared HTTP mode**: `python server.py --http` starts and answers on localhost, but hasn't been tested as a shared service for several developers.

## Setup

Put this folder in the repository (for example `tools/repo-search/`), then:

```sh
pip install -r tools/repo-search/requirements.txt
python tools/repo-search/repo_search.py index
```

The first run downloads the model from Hugging Face and the tree-sitter grammars for the languages in the repo. The index is written to `.repo-search/index.db` at the repository root; that folder ignores itself, so nothing needs adding to `.gitignore`. Indexing commits every 64 chunks, so an interrupted run resumes.

Check retrieval from the terminal before wiring up an agent:

```sh
python tools/repo-search/repo_search.py search "how are failed requests retried?"
```

## Register with Claude Code

Commit a `.mcp.json` at the repository root so every developer gets the server:

```json
{
  "mcpServers": {
    "repo-search": {
      "type": "stdio",
      "command": "python",
      "args": ["tools/repo-search/server.py"]
    }
  }
}
```

or run `claude mcp add --scope project repo-search -- python tools/repo-search/server.py`, which writes the same file. `python` must be the interpreter that has the requirements installed, and the relative path assumes Claude Code is started at the repository root. Docs: https://docs.claude.com/en/docs/claude-code/mcp

You may also find it useful to add the following to your `.claude/settings.local.json`:

``` json
{
  "enabledMcpjsonServers": [
    "repo-search"
  ]
}
```

Any other MCP client can launch `python server.py` the same way. `python server.py --http` serves the same tools at `http://127.0.0.1:8000/mcp`.

## How it works

- **Files**: everything Git tracks, plus new files that are not ignored. Binaries, lock files, minified files, files over 1 MB and generated .NET XML documentation (`<DocumentationFile>` output, which repeats the source's `///` comments) are skipped.
- **Chunks**: files are parsed with tree-sitter and split along syntax boundaries into pieces of up to about 4,000 characters. A large class is split into its methods; small neighbours are merged. Markdown is split by heading instead, one section per chunk of up to about 1,500 characters, so a question about one section isn't diluted by its neighbours. Files in a language without a grammar are split on line boundaries. Changing these rules bumps `CHUNKER_VERSION`, which rebuilds the index on next start.
- **Vectors**: chunks are embedded with the `Document` prompt and queries with `SearchQuery`, at 256 dimensions, and stored in SQLite. Search is exact cosine similarity in memory.
- **Freshness**: `search_code` checks for changed files at most every 30 seconds and re-embeds them first. More than 100 changed files (a branch switch) is left for an explicit `reindex`, and the agent is told the index is stale.

## Settings

Environment variables, all optional (set them under `"env"` in `.mcp.json`):

| Variable | Default |
| --- | --- |
| `REPO_SEARCH_REPO` | directory the server starts in |
| `REPO_SEARCH_DB` | `<repo>/.repo-search/index.db` |
| `REPO_SEARCH_MODEL` | `google/embeddinggemma-2` |
| `REPO_SEARCH_DIM` | `256` (also 768, 512, 128) |
| `REPO_SEARCH_QUERY_PROMPT` | `SearchQuery` |
| `REPO_SEARCH_DOC_PROMPT` | `Document` |
| `REPO_SEARCH_AUTO_REFRESH` | `1` (`0` disables the check in `search_code`) |
| `REPO_SEARCH_HOST`, `REPO_SEARCH_PORT` | `127.0.0.1`, `8000` (HTTP mode) |

Changing the model, dimension or document prompt clears the index, because the old vectors are no longer comparable. To see which prompt names the model ships:

```sh
python -c "from sentence_transformers import SentenceTransformer as S; print(S('google/embeddinggemma-2', config_kwargs={'vision_config': None, 'audio_config': None}).prompts)"
```
