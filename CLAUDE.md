# CLAUDE.md

## Searching this codebase

This repository has a semantic code search MCP server, `repo-search` (see `tools/repo-search/README.md`).

- For conceptual questions about the codebase ("where/how/why is X done"), call `mcp__repo-search__search_code` first, then confirm with Read before answering, citing paths and line numbers.
- For exact identifiers, strings, and finding callers, use Grep.
- If `repo-search` is not connected or reports an empty or stale index, fall back to Grep/Glob and mention that the index needs `reindex`.
