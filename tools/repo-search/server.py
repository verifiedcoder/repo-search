"""MCP server that lets an agent ask questions about a Git repository.

Runs over stdio by default (one process per developer, against their own checkout):

    python server.py

or as an HTTP service at http://$REPO_SEARCH_HOST:$REPO_SEARCH_PORT/mcp (default 127.0.0.1:8000):

    python server.py --http

The repository is $REPO_SEARCH_REPO, or the directory the server is started in.
See repo_search.py for the other settings. Requires the `mcp` package, version 2.x.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

from mcp.server.mcpserver import MCPServer

from repo_search import Config, Index

AUTO_REFRESH = os.environ.get("REPO_SEARCH_AUTO_REFRESH", "1") != "0"
AUTO_REFRESH_INTERVAL = 30      # seconds between staleness checks
AUTO_REFRESH_MAX_FILES = 100    # more changes than this (a branch switch) needs an explicit reindex
SNIPPET_CHARS = 2000
READ_MAX_LINES = 400

mcp = MCPServer(
    "repo-search",
    instructions=(
        "Tools for answering questions about this Git repository. Start with search_code for "
        "conceptual questions ('where is X handled', 'how does Y work'). Use grep_code when you "
        "know an exact identifier or string, and to find callers and usages. Use read_file to "
        "see the code around a hit before answering, and cite paths and line numbers."
    ),
)

_index: Index | None = None
_index_lock = threading.Lock()
_last_check = 0.0


def index() -> Index:
    global _index
    with _index_lock:
        if _index is None:
            _index = Index(Config.from_env())
        return _index


def _refresh() -> str:
    """Keep the index current between searches. Returns a note for the agent, or ''."""
    global _last_check
    if not AUTO_REFRESH or time.monotonic() - _last_check < AUTO_REFRESH_INTERVAL:
        return ""
    _last_check = time.monotonic()
    pending = index().pending()
    if not len(pending):
        return ""
    if len(pending) > AUTO_REFRESH_MAX_FILES:
        return (f"Note: the index is out of date ({len(pending)} files changed). "
                "Results may be stale; call reindex to update.\n\n")
    index().update(pending=pending)
    return ""


@mcp.tool()
def search_code(query: str, k: int = 8, path_glob: str | None = None) -> str:
    """Semantic search over the repository's code and docs.

    Finds code by meaning, so describe what you are looking for in plain language
    ("retry logic for failed uploads", "where the session cookie is validated").
    For an exact identifier or string, grep_code is more reliable.

    Args:
        query: What to look for, as a question or description.
        k: Number of results to return (default 8).
        path_glob: Optional filter such as "src/api/*" or "*.py" (* also matches "/").
    """
    idx = index()
    if idx.count()[1] == 0:
        return ("The index is empty. Call reindex to build it, or run "
                "`python repo_search.py index` in a terminal (the first build of a large "
                "repository can take a while and is resumable).")
    note = _refresh()
    hits = idx.search(query, k=min(max(k, 1), 30), path_glob=path_glob)
    if not hits:
        return note + "No results."
    parts = []
    for h in hits:
        text = h.text
        if len(text) > SNIPPET_CHARS:
            text = text[:SNIPPET_CHARS] + "\n[... truncated; use read_file for the rest]"
        parts.append(f"=== {h.path}:{h.start_line}-{h.end_line}  (score {h.score:.3f})\n{text}")
    return note + "\n\n".join(parts)


@mcp.tool()
def grep_code(pattern: str, path_glob: str | None = None, ignore_case: bool = False,
              max_results: int = 50) -> str:
    """Exact search with an extended regular expression (git grep).

    Use for identifiers, error messages, config keys, and to find every caller or
    usage of something search_code turned up.

    Args:
        pattern: Extended regex, e.g. "refreshToken\\(" or "TODO|FIXME".
        path_glob: Optional git pathspec such as "src/" or "*.ts".
        ignore_case: Case-insensitive match.
        max_results: Maximum matching lines to return (default 50).
    """
    cmd = ["git", "-C", str(index().cfg.repo), "grep", "-n", "-I", "-E", "--untracked"]
    if ignore_case:
        cmd.append("-i")
    cmd += ["-e", pattern, "--"]
    if path_glob:
        cmd.append(path_glob)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    if result.returncode == 1:
        return "No matches."
    if result.returncode != 0:
        return f"git grep failed: {result.stderr.strip()}"
    lines = result.stdout.splitlines()
    shown = [ln if len(ln) <= 300 else ln[:300] + " [...]" for ln in lines[:max(1, max_results)]]
    if len(lines) > len(shown):
        shown.append(f"[{len(lines) - len(shown)} more matches not shown; narrow the pattern "
                     "or path_glob]")
    return "\n".join(shown)


@mcp.tool()
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read part of a file in the repository, with line numbers.

    Args:
        path: Path relative to the repository root, as returned by the search tools.
        start_line: First line to return (1-based).
        end_line: Last line to return (default: start_line + 399).
    """
    repo = index().cfg.repo
    full = (repo / path).resolve()
    if not full.is_relative_to(repo):
        return "Refused: that path is outside the repository."
    if not full.is_file():
        return f"No such file: {path}"
    lines = full.read_bytes().decode("utf-8", errors="replace").split("\n")
    start = max(1, start_line)
    end = min(len(lines), end_line or start + READ_MAX_LINES - 1, start + READ_MAX_LINES - 1)
    body = "\n".join(f"{i:>6}  {lines[i - 1].rstrip(chr(13))}" for i in range(start, end + 1))
    return f"{path} (lines {start}-{end} of {len(lines)})\n{body}"


@mcp.tool()
def reindex(full: bool = False) -> str:
    """Update the semantic index to match the working tree.

    Only files that changed since the last run are re-embedded. Small changes are
    picked up automatically by search_code, so this is mainly for the first build
    and after switching branches.

    Args:
        full: Discard the index and rebuild everything.
    """
    stats = index().update(full=full)
    return (f"Index is current at {stats['head']}: {stats['files_indexed']} files, "
            f"{stats['chunks_indexed']} chunks ({stats['files_updated']} files re-embedded, "
            f"{stats['files_removed']} removed).")


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run("streamable-http",
                host=os.environ.get("REPO_SEARCH_HOST", "127.0.0.1"),
                port=int(os.environ.get("REPO_SEARCH_PORT", "8000")))
    else:
        mcp.run("stdio")
