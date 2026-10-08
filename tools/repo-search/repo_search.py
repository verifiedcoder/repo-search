"""Semantic index over a Git repository, built on EmbeddingGemma 2.

Library used by server.py, plus a small CLI:

    python repo_search.py index        # build or incrementally update the index
    python repo_search.py index --full # throw the index away and rebuild
    python repo_search.py search "where do we refresh auth tokens?"

Configuration is by environment variable (all optional):

    REPO_SEARCH_REPO          repository path            (default: current directory)
    REPO_SEARCH_DB            index file                 (default: <repo>/.repo-search/index.db)
    REPO_SEARCH_MODEL         embedding model            (default: google/embeddinggemma-2)
    REPO_SEARCH_DIM           embedding dimensions       (default: 256; one of 768/512/256/128)
    REPO_SEARCH_QUERY_PROMPT  prompt_name for queries    (default: SearchQuery)
    REPO_SEARCH_DOC_PROMPT    prompt_name for documents  (default: Document)
"""

from __future__ import annotations

import argparse
import bisect
import fnmatch
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_CHARS = 4000          # target upper bound for one chunk (~1k tokens; the model takes 8,192)
MAX_PROSE_CHARS = 1500    # Markdown: one section per chunk, so a question matches its section
MIN_CHARS = 200           # chunks smaller than this are merged into a neighbour
MAX_FILE_BYTES = 1_000_000
EMBED_BATCH = 64          # chunks embedded (and committed) per batch, so indexing is resumable
CHUNKER_VERSION = 2       # bump when chunking rules change; part of the index signature
SKIP_NAMES = {"package-lock.json", "pnpm-lock.yaml", "go.sum"}
SKIP_SUFFIXES = (".lock", ".min.js", ".min.css", ".map", ".svg")
MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")
# .NET XML documentation output (<DocumentationFile>): duplicates the /// comments in the source.
_DOTNET_XMLDOC = re.compile(rb"\A(?:\xef\xbb\xbf)?\s*(?:<\?xml[^>]*\?>\s*)?<doc>\s*<assembly>")


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    repo: Path
    db: Path
    model: str
    dim: int
    query_prompt: str
    doc_prompt: str

    @classmethod
    def from_env(cls, repo: str | os.PathLike | None = None) -> "Config":
        start = Path(repo or os.environ.get("REPO_SEARCH_REPO") or ".").resolve()
        root = Path(git(start, "rev-parse", "--show-toplevel").strip()).resolve()
        return cls(
            repo=root,
            db=Path(os.environ.get("REPO_SEARCH_DB") or root / ".repo-search" / "index.db"),
            model=os.environ.get("REPO_SEARCH_MODEL", "google/embeddinggemma-2"),
            dim=int(os.environ.get("REPO_SEARCH_DIM", "256")),
            query_prompt=os.environ.get("REPO_SEARCH_QUERY_PROMPT", "SearchQuery"),
            doc_prompt=os.environ.get("REPO_SEARCH_DOC_PROMPT", "Document"),
        )


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout


# --------------------------------------------------------------------------- chunking

_parsers: dict[str, object | None] = {}


def _parser_for(path: str):
    """Tree-sitter parser for this file's language, or None (grammars download on first use)."""
    try:
        import tree_sitter_language_pack as tslp
    except ImportError:
        return None
    lang = tslp.detect_language_from_path(path)
    if not lang:
        return None
    if lang not in _parsers:
        try:
            _parsers[lang] = tslp.get_parser(lang)
        except Exception:
            _parsers[lang] = None
    return _parsers[lang]


def _node_spans(node, max_bytes: int) -> list[tuple[int, int]]:
    """Group a node's children into byte spans of at most max_bytes.

    Small siblings (imports, short functions) are merged; anything too large
    (a big class) is split by recursing into its own children (its methods).
    Language-agnostic: no per-grammar list of node types.
    """
    spans: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for child in node.children:
        s, e = child.start_byte, child.end_byte
        if e - s > max_bytes:
            if cur:
                spans.append(cur)
                cur = None
            spans.extend(_node_spans(child, max_bytes) if child.child_count else [(s, e)])
        elif cur and e - cur[0] <= max_bytes:
            cur = (cur[0], e)
        else:
            if cur:
                spans.append(cur)
            cur = (s, e)
    if cur:
        spans.append(cur)
    return spans


def _markdown_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """One (start, end) line range per heading section; '#' lines inside code fences don't count."""
    starts, fence = [1], None
    for i, ln in enumerate(lines, 1):
        s = ln.lstrip()
        if s.startswith(("```", "~~~")):
            if fence is None:
                fence = s[:3]
            elif s.startswith(fence):
                fence = None
        elif fence is None and i > 1 and re.match(r"#{1,6}(\s|$)", s) and len(ln) - len(s) < 4:
            starts.append(i)
    ends = [s - 1 for s in starts[1:]] + [len(lines)]
    return list(zip(starts, ends))


def is_generated(data: bytes) -> bool:
    """Build output that is committed but adds nothing beyond the source it came from."""
    return _DOTNET_XMLDOC.match(data[:512]) is not None


def chunk_file(path: str, data: bytes) -> list[tuple[int, int, str]]:
    """Split a file into (start_line, end_line, text) chunks; lines are 1-based, inclusive."""
    lines = [ln.rstrip("\r") for ln in data.decode("utf-8", errors="replace").split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []
    n = len(lines)
    size = [0]                                  # prefix sums of line lengths (chars)
    for ln in lines:
        size.append(size[-1] + len(ln) + 1)

    def chars(a: int, b: int) -> int:           # chars in lines a..b inclusive
        return size[b] - size[a - 1]

    # 1. Structural ranges: heading sections for Markdown, the syntax tree for code,
    #    else the whole file as one range.
    ranges: list[tuple[int, int]] = []
    markdown = path.lower().endswith(MARKDOWN_SUFFIXES)
    limit = MAX_PROSE_CHARS if markdown else MAX_CHARS
    parser = None if markdown else _parser_for(path)
    if markdown:
        ranges = _markdown_ranges(lines)
    elif parser is not None:
        try:
            line_starts = [0]
            for raw in data.split(b"\n")[:-1]:
                line_starts.append(line_starts[-1] + len(raw) + 1)
            root = parser.parse(data).root_node
            prev_end = 0
            for s, e in _node_spans(root, MAX_CHARS):
                a = max(bisect.bisect_right(line_starts, s), prev_end + 1)
                b = min(bisect.bisect_right(line_starts, max(e - 1, s)), n)
                if a <= b:
                    ranges.append((a, b))
                    prev_end = b
        except Exception:
            ranges = []
    if not ranges:
        ranges = [(1, n)]

    # 2. Split anything still too large on line boundaries.
    split: list[tuple[int, int]] = []
    for a, b in ranges:
        start = a
        for i in range(a, b + 1):
            if i > start and chars(start, i) > limit:
                split.append((start, i - 1))
                start = i
        split.append((start, b))

    # 3. Fold fragments (a lone "class Foo:" header, a stray brace) into a neighbour.
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for a, b in split:
        if carry is not None:
            a, carry = carry, None
        if chars(a, b) < MIN_CHARS:
            carry = a
        else:
            merged.append((a, b))
    if carry is not None:
        last = split[-1][1]
        merged[-1:] = [(merged[-1][0] if merged else carry, last)]

    out = []
    for a, b in merged:
        text = "\n".join(lines[a - 1:b])[:MAX_CHARS * 4]   # guards against minified one-liners
        if text.strip():
            out.append((a, b, text))
    return out


# --------------------------------------------------------------------------- embedding

class Embedder:
    """Lazy wrapper around the model, so importing this module (and MCP startup) stays fast."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        from sentence_transformers import SentenceTransformer  # needs >= 6.1.0

        kwargs = {}
        if "embeddinggemma-2" in self.cfg.model.lower():
            # Text/code only: never load the vision and audio encoders (270M parameters).
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        return SentenceTransformer(self.cfg.model, truncate_dim=self.cfg.dim, **kwargs)

    def _encode(self, texts: list[str], prompt: str, progress: bool) -> np.ndarray:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            vecs = self._model.encode(
                texts, prompt_name=prompt, normalize_embeddings=True,
                batch_size=16, show_progress_bar=progress,
            )
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def documents(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.cfg.doc_prompt, progress)

    def query(self, text: str) -> np.ndarray:
        return self._encode([text], self.cfg.query_prompt, False)[0]


# --------------------------------------------------------------------------- index

@dataclass
class Hit:
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


@dataclass
class Pending:
    changed: dict[str, tuple[str, int, int]]    # path -> (hash, mtime_ns, size)
    touched: dict[str, tuple[int, int]]         # same content, new mtime: just refresh the stat
    removed: list[str]

    def __len__(self) -> int:
        return len(self.changed) + len(self.removed)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT NOT NULL,
                                 mtime INTEGER NOT NULL, size INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT NOT NULL,
                                  start_line INTEGER, end_line INTEGER,
                                  text TEXT, embedding BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""


class Index:
    def __init__(self, cfg: Config, embedder: Embedder | None = None):
        self.cfg = cfg
        self.embedder = embedder or Embedder(cfg)
        self._lock = threading.RLock()
        self._matrix: tuple[str, list[tuple], np.ndarray] | None = None
        self._ready = False

    # -- storage

    @contextmanager
    def _db(self):
        """A connection that commits on success and is always closed."""
        folder = self.cfg.db.parent
        if not folder.exists():
            folder.mkdir(parents=True)
            (folder / ".gitignore").write_text("*\n")   # the index never gets committed
        con = sqlite3.connect(self.cfg.db)
        try:
            with con:
                if not self._ready:
                    self._prepare(con)
                    self._ready = True
                yield con
        finally:
            con.close()

    def _prepare(self, con: sqlite3.Connection) -> None:
        con.executescript(SCHEMA)
        # Vectors from a different model, dimension or prompt are not comparable: start over.
        signature = f"{self.cfg.model}|{self.cfg.dim}|{self.cfg.doc_prompt}|chunker{CHUNKER_VERSION}"
        row = con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if row is None or row[0] != signature:
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM files")
            con.execute("INSERT OR REPLACE INTO meta VALUES('signature', ?)", (signature,))
            self._bump(con)

    @staticmethod
    def _bump(con: sqlite3.Connection) -> None:
        con.execute(
            "INSERT INTO meta VALUES('version', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )

    def count(self) -> tuple[int, int]:
        """(files, chunks) currently indexed."""
        with self._lock, self._db() as con:
            return (con.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                    con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # -- indexing

    def _candidates(self) -> list[str]:
        # Tracked files plus new files that aren't ignored, so work in progress is searchable.
        out = git(self.cfg.repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        paths = sorted({p for p in out.split("\0") if p})
        return [p for p in paths
                if not p.endswith(SKIP_SUFFIXES) and p.rsplit("/", 1)[-1] not in SKIP_NAMES]

    def pending(self) -> Pending:
        """What an update would do. Cheap: unchanged files are recognised by mtime + size."""
        with self._lock, self._db() as con:
            known = {p: (h, m, s) for p, h, m, s in con.execute("SELECT * FROM files")}
        result = Pending({}, {}, [])
        seen = set()
        for path in self._candidates():
            full = self.cfg.repo / path
            try:
                if full.is_symlink() or not full.is_file():
                    continue
                st = full.stat()
                if st.st_size > MAX_FILE_BYTES:
                    continue
                seen.add(path)
                old = known.get(path)
                if old and old[1] == st.st_mtime_ns and old[2] == st.st_size:
                    continue
                digest = hashlib.sha1(full.read_bytes()).hexdigest()
            except OSError:
                continue
            if old and old[0] == digest:
                result.touched[path] = (st.st_mtime_ns, st.st_size)
            else:
                result.changed[path] = (digest, st.st_mtime_ns, st.st_size)
        result.removed = [p for p in known if p not in seen]
        return result

    def update(self, full: bool = False, progress: bool = False,
               pending: Pending | None = None) -> dict:
        """Bring the index in line with the working tree. Only changed files are re-embedded."""
        with self._lock:
            if full:
                with self._db() as con:
                    con.execute("DELETE FROM chunks")
                    con.execute("DELETE FROM files")
                    self._bump(con)
                pending = None
            todo = pending or self.pending()

            with self._db() as con:
                for path in todo.removed:
                    con.execute("DELETE FROM chunks WHERE path=?", (path,))
                    con.execute("DELETE FROM files WHERE path=?", (path,))
                for path, (mtime, size) in todo.touched.items():
                    con.execute("UPDATE files SET mtime=?, size=? WHERE path=?", (mtime, size, path))
                if todo.removed:
                    self._bump(con)

            chunks_written = 0
            batch: list[tuple[str, tuple, list]] = []    # (path, file row, chunks)
            batch_chunks = 0
            items = list(todo.changed.items())
            for i, (path, (digest, mtime, size)) in enumerate(items):
                try:
                    data = (self.cfg.repo / path).read_bytes()
                except OSError:
                    continue
                skip = b"\0" in data[:8192] or is_generated(data)   # binaries, build output
                chunks = [] if skip else chunk_file(path, data)
                batch.append((path, (path, digest, mtime, size), chunks))
                batch_chunks += len(chunks)
                if batch_chunks >= EMBED_BATCH or i == len(items) - 1:
                    chunks_written += self._write(batch, progress)
                    batch, batch_chunks = [], 0
                    if progress:
                        print(f"  {i + 1}/{len(items)} files", file=sys.stderr)
            if batch:
                chunks_written += self._write(batch, progress)

            with self._db() as con:
                head = git(self.cfg.repo, "rev-parse", "--short", "HEAD").strip()
                con.execute("INSERT OR REPLACE INTO meta VALUES('head', ?)", (head,))
            files, total = self.count()
            return {"files_updated": len(todo.changed), "files_removed": len(todo.removed),
                    "chunks_written": chunks_written, "files_indexed": files,
                    "chunks_indexed": total, "head": head}

    def _write(self, batch: list, progress: bool) -> int:
        """Embed one batch and commit it atomically, so an interrupted run can resume."""
        texts = [f"File: {path}\n\n{text}" for path, _, chunks in batch for _, _, text in chunks]
        vecs = self.embedder.documents(texts, progress) if texts else np.zeros((0, self.cfg.dim))
        with self._db() as con:
            i = 0
            for path, file_row, chunks in batch:
                con.execute("DELETE FROM chunks WHERE path=?", (path,))
                for a, b, text in chunks:
                    con.execute(
                        "INSERT INTO chunks(path, start_line, end_line, text, embedding) "
                        "VALUES(?,?,?,?,?)",
                        (path, a, b, text, vecs[i].astype(np.float32).tobytes()),
                    )
                    i += 1
                con.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", file_row)
            self._bump(con)
        return len(texts)

    # -- search

    def _load_matrix(self, con: sqlite3.Connection):
        version = con.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0]
        if self._matrix is None or self._matrix[0] != version:
            rows = con.execute(
                "SELECT id, path, start_line, end_line, embedding FROM chunks").fetchall()
            blob = b"".join(r[4] for r in rows)
            matrix = (np.frombuffer(blob, dtype=np.float32).reshape(len(rows), -1)
                      if rows else np.zeros((0, self.cfg.dim), dtype=np.float32))
            self._matrix = (version, [r[:4] for r in rows], matrix)
        return self._matrix[1], self._matrix[2]

    def search(self, query: str, k: int = 8, path_glob: str | None = None) -> list[Hit]:
        """Top-k chunks by cosine similarity (exact search; fine up to ~1M chunks)."""
        with self._lock, self._db() as con:
            rows, matrix = self._load_matrix(con)
            if not rows:
                return []
            scores = matrix @ self.embedder.query(query)
            if path_glob:
                keep = np.fromiter((fnmatch.fnmatch(r[1], path_glob) for r in rows),
                                   dtype=bool, count=len(rows))
                scores = np.where(keep, scores, -np.inf)
            hits = []
            for i in np.argsort(-scores)[:max(1, k)]:
                if not np.isfinite(scores[i]):
                    break
                cid, path, a, b = rows[i]
                text = con.execute("SELECT text FROM chunks WHERE id=?", (cid,)).fetchone()[0]
                hits.append(Hit(path, a, b, float(scores[i]), text))
            return hits


# --------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic index over a Git repository.")
    ap.add_argument("--repo", help="repository path (default: $REPO_SEARCH_REPO or .)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_index = sub.add_parser("index", help="build or update the index")
    p_index.add_argument("--full", action="store_true", help="rebuild from scratch")
    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--path", help="only search paths matching this glob")
    args = ap.parse_args()

    index = Index(Config.from_env(args.repo))
    if args.cmd == "index":
        print(f"Indexing {index.cfg.repo} with {index.cfg.model} ({index.cfg.dim}d)",
              file=sys.stderr)
        print(index.update(full=args.full, progress=True))
    else:
        for h in index.search(args.query, args.k, args.path):
            print(f"\n=== {h.path}:{h.start_line}-{h.end_line}  ({h.score:.3f})\n{h.text}")


if __name__ == "__main__":
    main()
