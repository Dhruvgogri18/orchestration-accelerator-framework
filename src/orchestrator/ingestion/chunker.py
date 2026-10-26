"""Structure-aware chunking.

Instead of cutting files into fixed token windows, every file type is split
along its own logical units, so a chunk is one self-contained thing:

  * YAML / JSON  -> one chunk per catalog entry (a job), per resource entry
                    (an SNS topic) or per config section; tiny neighbours are merged
  * Python       -> the whole file when it fits, else groups of top-level
                    functions/classes with the imports + docstring repeated
                    (the cAST idea: split along the AST, merge siblings up to a
                    size limit)
  * Markdown     -> one chunk per heading section, fenced code never split,
                    tiny sections merged into their neighbour

Fixed token windows (CHUNK_SIZE_TOKENS / CHUNK_OVERLAP_TOKENS) are only the
fallback for a single unit that is too big, or a file that cannot be parsed.

Each chunk starts with a one-line header ("[repo] path | what this is") before
it is embedded -- the same idea as Anthropic's Contextual Retrieval, but built
deterministically from file structure instead of an LLM call per chunk.

Everything is deterministic: same input text -> same chunks, ids and hashes.
Descriptive metadata (resource names, env vars, AWS services, downstream
targets, ...) comes from parsers, never from an LLM.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import yaml

from orchestrator.ingestion.models import Chunk, MetadataValue, SourceFile
from orchestrator.ingestion.secrets import redact_secrets

DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 120
MIN_CHUNK_TOKENS = 60  # smaller config sections are merged into a neighbour
# Doc sections are merged only when they are little more than a heading: a short
# section with its own heading (e.g. "Alerting") is usually its own topic.
MIN_DOC_SECTION_TOKENS = 25
HEADER_RESERVE_TOKENS = 40  # room kept for the header inside max_tokens
MAX_MENTIONS = 30

TokenCounter = Callable[[str], int]

# Known config section names -> the AWS resource type their entries describe.
_SECTION_TYPES = {
    "buckets": "s3_bucket",
    "tables": "dynamodb_table",
    "sns_topics": "sns_topic",
    "topics": "sns_topic",
    "queues": "sqs_queue",
    "lambdas": "lambda",
    "lambda_functions": "lambda",
    "glue_jobs": "glue_job",
    "pipelines": "pipeline",
}
_PIPELINE_KEYS = {"pipelines", "workflows"}
_CATALOG_TYPE_TO_KIND = {"lambda": "lambda_handler", "glue": "glue_script"}


# ---------------------------------------------------------------- tokens / sizes
def _default_counter() -> TokenCounter:
    try:
        from orchestrator.api.services.llm_trace import count_tokens

        return count_tokens
    except Exception:  # pragma: no cover - llm_trace is part of this repo
        return lambda text: max(1, len(text) // 4) if text else 0


def _default_sizes() -> tuple[int, int]:
    try:
        from orchestrator.config import get_settings

        s = get_settings()
        return s.chunk_size_tokens, s.chunk_overlap_tokens
    except Exception:  # missing gateway settings must not block chunking
        return DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS


@dataclass(frozen=True)
class _Ctx:
    repo_id: str
    registry: Registry
    max_tokens: int
    overlap_tokens: int
    min_tokens: int
    count: TokenCounter
    min_doc_tokens: int = MIN_DOC_SECTION_TOKENS

    @property
    def budget(self) -> int:
        """Largest body size so that header + body still fit in max_tokens."""
        return max(self.max_tokens - HEADER_RESERVE_TOKENS, self.max_tokens // 2, 20)


# ---------------------------------------------------------------- YAML loading
class _Loader(yaml.SafeLoader):
    """SafeLoader that tolerates custom tags (CloudFormation's !Ref, !Sub, ...)."""


def _any_tag(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)


_Loader.add_multi_constructor("!", _any_tag)


def _load_data(source: SourceFile) -> Any:
    if source.file_type == "yaml":
        return yaml.load(source.text, Loader=_Loader)  # noqa: S506 - SafeLoader subclass
    return json.loads(source.text)


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class CatalogEntry:
    name: str
    type: str | None
    status: str
    script: str | None
    entry_point: str | None


def _resource_name(value: str) -> str:
    """Last component of an ARN, otherwise the value itself."""
    if value.startswith("arn:"):
        return re.split(r"[:/]", value)[-1]
    return value


def _names_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_resource_name(value)]
    if isinstance(value, list):
        return [
            n
            for item in value
            for n in _names_from(item.get("name") if isinstance(item, dict) else item)
        ]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                out.append(str(item.get("name") or key))
            else:
                out.extend(_names_from(item))
        return out
    return []


def _catalog_items(data: Any) -> list[dict]:
    """Entries of any list of dicts that look like job/function definitions."""
    items: list[dict] = []
    if isinstance(data, dict):
        for value in data.values():
            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(i, dict) and "name" in i and ("script" in i or "type" in i)
                    for i in value
                )
            ):
                items.extend(value)
    return items


class Registry:
    """What the repo says exists: job/function catalog + named resources.

    Built in a first pass over the whole repo so that a handler's chunk can be
    labelled with the catalog entry that points at its script, and so that any
    chunk can record which known resources it mentions.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, CatalogEntry] = {}
        self.names: dict[str, str] = {}  # resource name -> "active" | "deprecated" | ...
        self._pattern: re.Pattern[str] | None = None

    @classmethod
    def from_sources(cls, sources: list[SourceFile]) -> Registry:
        reg = cls()
        for source in sources:
            if source.file_type not in ("yaml", "json"):
                continue
            try:
                data = _load_data(source)
            except Exception:  # noqa: BLE001 - unparseable files are chunked as text
                continue
            for item in _catalog_items(data):
                entry = CatalogEntry(
                    name=str(item["name"]),
                    type=str(item["type"]) if item.get("type") else None,
                    status=str(item.get("status") or "active"),
                    script=str(item["script"]) if item.get("script") else None,
                    entry_point=str(item["entry_point"]) if item.get("entry_point") else None,
                )
                reg.names[entry.name] = entry.status
                if entry.script:
                    reg.scripts[entry.script] = entry
            if isinstance(data, dict) and isinstance(data.get("environment"), str):
                for key, value in data.items():
                    if key not in ("environment", "aws"):
                        for name in _names_from(value):
                            reg.names.setdefault(name, "active")
        return reg

    def mentions(self, text: str) -> list[str]:
        if self._pattern is None:
            names = sorted((n for n in self.names if len(n) >= 4), key=lambda n: (-len(n), n))
            self._pattern = (
                re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w-])")
                if names
                else re.compile(r"(?!x)x")
            )
        return sorted(set(self._pattern.findall(text)))[:MAX_MENTIONS]


# ---------------------------------------------------------------- units
@dataclass
class _Unit:
    key: str  # stable id component, e.g. "jobs:csv-transform-and-load"
    kind: str  # chunk_kind metadata
    title: str  # header text after the file path
    body: str
    meta: dict[str, Any] = field(default_factory=dict)
    start_line: int | None = None  # 1-based, inclusive
    end_line: int | None = None
    part: int | None = None  # set when a splitter cut one logical unit into several
    of: int | None = None


def _window_split(
    text: str, budget: int, overlap: int, count: TokenCounter
) -> list[tuple[str, int, int]]:
    """Fallback splitter: line windows <= budget tokens with ~overlap tokens of
    overlap. Returns (text, first_line_index, last_line_index) per window; lines
    longer than the budget are cut so a window can never exceed it."""
    pieces: list[tuple[str, int]] = []  # (text, original line index)
    for idx, line in enumerate(text.split("\n")):
        rest = line
        while count(rest) > budget and len(rest) > 1:
            cut = max(1, min(len(rest) // 2, budget * 3))
            while cut > 1 and count(rest[:cut]) > budget:
                cut //= 2
            pieces.append((rest[:cut], idx))
            rest = rest[cut:]
        pieces.append((rest, idx))
    tok = [count(p) + 1 for p, _ in pieces]

    windows: list[tuple[str, int, int]] = []
    i = 0
    while i < len(pieces):
        j, total = i, 0
        while j < len(pieces) and (j == i or total + tok[j] <= budget):
            total += tok[j]
            j += 1
        windows.append(("\n".join(p for p, _ in pieces[i:j]), pieces[i][1], pieces[j - 1][1]))
        if j >= len(pieces):
            break
        k, back = j, 0
        while k - 1 > i and back + tok[k - 1] <= overlap:
            k -= 1
            back += tok[k]
        i = k
    return windows


def _text_units(source: SourceFile, ctx: _Ctx, kind: str = "text_window") -> list[_Unit]:
    windows = _window_split(source.text, ctx.budget, ctx.overlap_tokens, ctx.count)
    return [
        _Unit(
            key=f"window-{n}",
            kind=kind,
            title=f"{PurePosixPath(source.path).name} lines {s + 1}-{e + 1}",
            body=text,
            start_line=s + 1,
            end_line=e + 1,
        )
        for n, (text, s, e) in enumerate(windows, start=1)
    ]


# ---------------------------------------------------------------- Python
def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _first_str_arg(call: ast.Call, index: int = 0) -> str | None:
    if len(call.args) > index and isinstance(call.args[index], ast.Constant):
        value = call.args[index].value
        return value if isinstance(value, str) else None
    return None


def _python_facts(tree: ast.AST) -> dict[str, list[str]]:
    env: set[str] = set()
    services: set[str] = set()
    glue_args: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in {"os.environ.get", "os.getenv", "environ.get", "getenv"} or name.endswith(
                "require_env"
            ):
                if (arg := _first_str_arg(node)) is not None:
                    env.add(arg)
            elif (
                name == "boto3.client"
                or name == "boto3.resource"
                or name.endswith((".client", ".resource"))
            ):
                if (arg := _first_str_arg(node)) is not None:
                    services.add(arg)
            elif name.endswith("getResolvedOptions") and len(node.args) > 1:
                names = node.args[1]
                if isinstance(names, ast.List):
                    glue_args.update(
                        e.value
                        for e in names.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
        elif isinstance(node, ast.Subscript) and _dotted(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                env.add(node.slice.value)
    glue_args.discard("JOB_NAME")
    top = [
        n.name
        for n in getattr(tree, "body", [])
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return {
        "env_vars": sorted(env),
        "aws_services": sorted(services),
        "glue_args": sorted(glue_args),
        "functions": sorted(top),
    }


def _span(node: ast.stmt) -> tuple[int, int]:
    start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    return start, node.end_lineno or node.lineno


def _python_units(source: SourceFile, ctx: _Ctx) -> list[_Unit]:
    try:
        tree = ast.parse(source.text)
    except SyntaxError:
        return _text_units(source, ctx, kind="python_module")

    entry = ctx.registry.scripts.get(source.path)
    facts = _python_facts(tree)
    docstring = ast.get_docstring(tree) or ""
    status = (
        entry.status
        if entry
        else ("deprecated" if re.search(r"\bDEPRECATED\b", docstring) else None)
    )
    rtype = entry.type if entry else None
    kind = _CATALOG_TYPE_TO_KIND.get(rtype or "", "python_module")
    detected_entry = (
        "lambda_handler"
        if "lambda_handler" in facts["functions"]
        else ("main" if "main" in facts["functions"] else None)
    )
    base_meta: dict[str, Any] = {
        "resource_name": entry.name if entry else None,
        "resource_type": rtype,
        "status": status,
        "entry_point": (entry.entry_point if entry and entry.entry_point else detected_entry),
        "env_vars": facts["env_vars"],
        "aws_services": facts["aws_services"],
        "glue_args": facts["glue_args"],
    }
    title = (
        f"{rtype or 'python'} {entry.name}"
        if entry
        else f"python module {PurePosixPath(source.path).name}"
    )

    if ctx.count(source.text) <= ctx.budget:
        return [
            _Unit(
                key="file",
                kind=kind,
                title=title,
                body=source.text.rstrip("\n"),
                meta={**base_meta, "functions": facts["functions"]},
                start_line=1,
                end_line=source.text.count("\n") + 1,
            )
        ]

    # Too big for one chunk: split along top-level statements (cAST-style) --
    # group consecutive definitions while the *composed* chunk (preamble + group)
    # still fits, so a chunk never needs a second, blind split.
    lines = source.text.split("\n")
    preamble_nodes = [
        n
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Import | ast.ImportFrom)
        or (i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    pre_ids = {id(n) for n in preamble_nodes}
    preamble = "\n".join("\n".join(lines[s - 1 : e]) for s, e in map(_span, preamble_nodes))

    def node_text(nodes: list[ast.stmt]) -> str:
        return "\n".join(lines[_span(nodes[0])[0] - 1 : _span(nodes[-1])[1]])

    def compose(text: str) -> str:
        return f"{preamble}\n\n{text}" if preamble else text

    room = max(ctx.budget - ctx.count(preamble) - 10, ctx.budget // 2)  # for an oversize node
    groups: list[tuple[list[ast.stmt], bool]] = []  # (nodes, needs_window_split)
    current: list[ast.stmt] = []
    for node in (n for n in tree.body if id(n) not in pre_ids):
        if ctx.count(compose(node_text([node]))) > ctx.budget:  # one definition is too big
            if current:
                groups.append((current, False))
                current = []
            groups.append(([node], True))
            continue
        if current and ctx.count(compose(node_text([*current, node]))) > ctx.budget:
            groups.append((current, False))
            current = []
        current.append(node)
    if current:
        groups.append((current, False))

    units: list[_Unit] = []
    for n, (group, needs_split) in enumerate(groups, start=1):
        first, last = _span(group[0])[0], _span(group[-1])[1]
        body = node_text(group)
        names = sorted(
            g.name
            for g in group
            if isinstance(g, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
        parts = (
            _window_split(body, room, ctx.overlap_tokens, ctx.count)
            if needs_split
            else [(body, 0, last - first)]
        )
        for p, (text, ps, pe) in enumerate(parts, start=1):
            units.append(
                _Unit(
                    key=f"part-{n}" + (f"-{p}" if len(parts) > 1 else ""),
                    kind=kind,
                    title=f"{title} (defines {', '.join(names) or 'module code'})",
                    body=compose(text),
                    meta={**base_meta, "functions": names},
                    start_line=first + ps,
                    end_line=first + pe,
                )
            )
    for i, unit in enumerate(units, start=1):
        unit.part, unit.of = i, len(units)
    return units


# ---------------------------------------------------------------- YAML / JSON
@dataclass
class _Piece:
    key: str
    data: Any
    text: str
    start: int | None = None
    end: int | None = None
    entries: list[_Piece] = field(default_factory=list)
    entries_kind: str = ""  # "seq" | "map" | ""


def _item_key(data: Any, index: int) -> str:
    if isinstance(data, dict) and data.get("name"):
        return str(data["name"])
    return f"[{index}]"


def _slice(lines: list[str], start: yaml.Mark, end: yaml.Mark) -> tuple[str, int, int]:
    """Whole-line text of a YAML node, with its 1-based line range."""
    s = start.line
    before_end = lines[end.line][: end.column] if end.line < len(lines) else ""
    e = end.line if (end.column == 0 or not before_end.strip()) else end.line + 1
    e = max(e, s + 1)
    block = lines[s:e]
    while len(block) > 1 and not block[-1].strip():
        block.pop()
    return textwrap.dedent("\n".join(block)), s + 1, s + len(block)


def _yaml_pieces(text: str) -> tuple[str, Any, list[_Piece]] | None:
    loader = _Loader(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        lines = text.split("\n")

        def build(node: yaml.Node) -> Any:
            return loader.construct_object(node, deep=True)

        data = build(root)
        if isinstance(root, yaml.SequenceNode):
            pieces = []
            for i, item in enumerate(root.value):
                t, s, e = _slice(lines, item.start_mark, item.end_mark)
                pieces.append(_Piece(_item_key(build(item), i), build(item), t, s, e))
            return "seq", data, pieces
        if not isinstance(root, yaml.MappingNode):
            return None
        pieces = []
        for knode, vnode in root.value:
            t, s, e = _slice(lines, knode.start_mark, vnode.end_mark)
            piece = _Piece(str(knode.value), build(vnode), t, s, e)
            if (
                isinstance(vnode, yaml.SequenceNode)
                and vnode.value
                and all(isinstance(i, yaml.MappingNode) for i in vnode.value)
            ):
                piece.entries_kind = "seq"
                for i, item in enumerate(vnode.value):
                    it, isl, iel = _slice(lines, item.start_mark, item.end_mark)
                    piece.entries.append(
                        _Piece(_item_key(build(item), i), build(item), it, isl, iel)
                    )
            elif (
                isinstance(vnode, yaml.MappingNode)
                and vnode.value
                and all(isinstance(v2, yaml.MappingNode) for _, v2 in vnode.value)
            ):
                piece.entries_kind = "map"
                for k2, v2 in vnode.value:
                    it, isl, iel = _slice(lines, k2.start_mark, v2.end_mark)
                    piece.entries.append(_Piece(str(k2.value), build(v2), it, isl, iel))
            pieces.append(piece)
        return "map", data, pieces
    except yaml.YAMLError:
        return None
    finally:
        loader.dispose()


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _json_pieces(text: str) -> tuple[str, Any, list[_Piece]] | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if isinstance(data, list):
        return (
            "seq",
            data,
            [_Piece(_item_key(item, i), item, _dump(item)) for i, item in enumerate(data)],
        )
    if not isinstance(data, dict):
        return None
    pieces: list[_Piece] = []
    for key, value in data.items():
        piece = _Piece(str(key), value, _dump({key: value}))
        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
            piece.entries_kind = "seq"
            piece.entries = [_Piece(_item_key(i, n), i, _dump(i)) for n, i in enumerate(value)]
        elif isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            piece.entries_kind = "map"
            piece.entries = [_Piece(str(k), v, _dump({k: v})) for k, v in value.items()]
        pieces.append(piece)
    return "map", data, pieces


def _str_list(value: Any) -> list[str] | None:
    return [str(v) for v in value] if isinstance(value, list) else None


def _item_meta(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    env = item.get("environment")
    args = item.get("arguments")
    return {
        "resource_type": item.get("type") if isinstance(item.get("type"), str) else None,
        "status": item.get("status") if isinstance(item.get("status"), str) else None,
        "script_path": item.get("script") if isinstance(item.get("script"), str) else None,
        "entry_point": item.get("entry_point")
        if isinstance(item.get("entry_point"), str)
        else None,
        "reads": _str_list(item.get("reads")),
        "writes": _str_list(item.get("writes")),
        "downstream": _str_list(item.get("downstream")),
        "env_vars": sorted(env) if isinstance(env, dict) else None,
        "glue_args": sorted(str(k).lstrip("-") for k in args) if isinstance(args, dict) else None,
    }


def _structured_units(root_kind: str, root: Any, pieces: list[_Piece], ctx: _Ctx) -> list[_Unit]:
    env = root.get("environment") if isinstance(root, dict) else None
    env = env if isinstance(env, str) else None
    units: list[_Unit] = []
    tiny: list[_Piece] = []

    def section_unit(group: list[_Piece]) -> _Unit:
        types = sorted({t for p in group if (t := _SECTION_TYPES.get(p.key))})
        names = sorted(
            {n for p in group if p.key not in ("environment", "aws") for n in _names_from(p.data)}
        )
        keys = "+".join(p.key for p in group)
        return _Unit(
            key=f"section:{keys}",
            kind="env_resources" if env else "config_section",
            title=f"{'environment ' + env + ' ' if env else ''}config section: {keys}",
            body="\n".join(p.text for p in group),
            meta={
                "environment": env,
                "resource_types": types,
                "resource_names": names if env else None,
            },
            start_line=group[0].start,
            end_line=group[-1].end,
        )

    def flush() -> None:
        nonlocal tiny
        if tiny:
            units.append(section_unit(tiny))
            tiny = []

    for piece in pieces:
        if root_kind == "seq":  # top-level list: every item is a unit
            data = piece.data
            name = piece.key
            units.append(
                _Unit(
                    key=f"item:{name}",
                    kind="config_entry",
                    title=f"config entry: {name}",
                    body=piece.text,
                    meta={
                        "resource_name": name
                        if isinstance(data, dict) and data.get("name")
                        else None,
                        **_item_meta(data),
                    },
                    start_line=piece.start,
                    end_line=piece.end,
                )
            )
            continue
        if piece.entries:
            flush()
            for entry in piece.entries:
                data = entry.data
                if piece.entries_kind == "seq":
                    name = entry.key
                    is_pipeline = piece.key in _PIPELINE_KEYS
                    is_job = isinstance(data, dict) and ("script" in data or piece.key == "jobs")
                    kind = (
                        "pipeline"
                        if is_pipeline
                        else "job_definition"
                        if is_job
                        else "config_entry"
                    )
                    rtype = _item_meta(data).get("resource_type") or (
                        "pipeline" if is_pipeline else _SECTION_TYPES.get(piece.key)
                    )
                    title = f"{piece.key} entry: {name}" + (
                        f" ({rtype})" if rtype and not is_pipeline else ""
                    )
                    meta = {
                        "environment": env,
                        "resource_name": name,
                        "resource_names": [name],
                        **_item_meta(data),
                    }
                    meta["resource_type"] = rtype
                else:  # mapping of mappings, e.g. sns_topics.ops_alerts
                    name = (
                        str(data.get("name") or entry.key) if isinstance(data, dict) else entry.key
                    )
                    kind = "env_resources" if env else "config_entry"
                    rtype = _SECTION_TYPES.get(piece.key)
                    title = f"{'environment ' + env + ' ' if env else ''}{piece.key}: {name}"
                    meta = {
                        "environment": env,
                        "resource_name": name,
                        "resource_names": [name],
                        **_item_meta(data),
                    }
                    meta["resource_type"] = _item_meta(data).get("resource_type") or rtype
                units.append(
                    _Unit(
                        key=f"{piece.key}:{entry.key}",
                        kind=kind,
                        title=title,
                        body=entry.text,
                        meta=meta,
                        start_line=entry.start,
                        end_line=entry.end,
                    )
                )
            continue
        if ctx.count(piece.text) < ctx.min_tokens:
            tiny.append(piece)
            if ctx.count("\n".join(p.text for p in tiny)) >= ctx.min_tokens:
                flush()
        else:
            flush()
            units.append(section_unit([piece]))
    # A leftover group of tiny sections joins the previous section chunk if it fits.
    if tiny:
        last = units[-1] if units else None
        merged_text = "\n".join(p.text for p in tiny)
        if (
            last
            and last.key.startswith("section:")
            and ctx.count(last.body + "\n" + merged_text) <= ctx.budget
        ):
            extra = section_unit(tiny)
            last.key += "+" + extra.key.removeprefix("section:")
            last.title += " + " + ", ".join(p.key for p in tiny)
            last.body += "\n" + merged_text
            last.end_line = extra.end_line
            last.meta["resource_types"] = sorted(
                {
                    *(last.meta.get("resource_types") or []),
                    *(extra.meta.get("resource_types") or []),
                }
            )
            if env:
                last.meta["resource_names"] = sorted(
                    {
                        *(last.meta.get("resource_names") or []),
                        *(extra.meta.get("resource_names") or []),
                    }
                )
        else:
            flush()
    return units


def _structured_file_units(source: SourceFile, ctx: _Ctx) -> list[_Unit]:
    parsed = _yaml_pieces(source.text) if source.file_type == "yaml" else _json_pieces(source.text)
    if parsed is None:
        return _text_units(source, ctx, kind=f"{source.file_type}_text")
    root_kind, root, pieces = parsed
    units = _structured_units(root_kind, root, pieces, ctx)
    return units or _text_units(source, ctx, kind=f"{source.file_type}_text")


# ---------------------------------------------------------------- Markdown
_HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*#*\s*$")


@dataclass
class _Section:
    path: str
    start: int  # 0-based line index
    lines: list[str]


def _md_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []
    current: _Section | None = None
    fence: str | None = None
    for idx, line in enumerate(text.split("\n")):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        match = None if fence else _HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [*stack[: level - 1], match.group(2)]
            current = _Section(" > ".join(stack), idx, [line])
            sections.append(current)
        else:
            if current is None:
                current = _Section("(top of file)", idx, [])
                sections.append(current)
            current.lines.append(line)
    return [s for s in sections if "\n".join(s.lines).strip()]


def _md_blocks(lines: list[str]) -> list[list[str]]:
    """Paragraph-level blocks; a fenced code block is always one block."""
    blocks: list[list[str]] = []
    current: list[str] = []
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = marker if fence is None else (None if marker == fence else fence)
        current.append(line)
        if fence is None and not line.strip():
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _markdown_units(source: SourceFile, ctx: _Ctx) -> list[_Unit]:
    sections = _md_sections(source.text)
    # Merge tiny sections into the NEXT one (a heading + its first child), else the previous.
    merged: list[_Section] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        if ctx.count("\n".join(sec.lines)) < ctx.min_doc_tokens:
            if i + 1 < len(sections):
                nxt = sections[i + 1]
                sections[i + 1] = _Section(nxt.path, sec.start, [*sec.lines, *nxt.lines])
                i += 1
                continue
            if merged:
                merged[-1].lines.extend(sec.lines)
                i += 1
                continue
        merged.append(sec)
        i += 1

    units: list[_Unit] = []
    for sec in merged:
        body_lines = list(sec.lines)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        body = "\n".join(body_lines)
        if ctx.count(body) <= ctx.budget:
            parts = [(body, 0, len(body_lines) - 1)]
        else:
            parts, buf, buf_start, offset = [], [], 0, 0
            for block in _md_blocks(body_lines):
                text = "\n".join(block)
                if buf and ctx.count("\n".join(buf + block)) > ctx.budget:
                    parts.append(("\n".join(buf).rstrip(), buf_start, offset - 1))
                    buf, buf_start = [], offset
                if ctx.count(text) > ctx.budget:
                    if buf:
                        parts.append(("\n".join(buf).rstrip(), buf_start, offset - 1))
                        buf = []
                    for wtext, ws, we in _window_split(
                        text, ctx.budget, ctx.overlap_tokens, ctx.count
                    ):
                        parts.append((wtext, offset + ws, offset + we))
                    buf_start = offset + len(block)
                else:
                    buf.extend(block)
                offset += len(block)
            if buf:
                parts.append(("\n".join(buf).rstrip(), buf_start, offset - 1))
        for n, (text, ps, pe) in enumerate(parts, start=1):
            units.append(
                _Unit(
                    key=f"section:{sec.path}" + (f"#{n}" if len(parts) > 1 else ""),
                    kind="doc_section",
                    title=sec.path,
                    body=text,
                    meta={"heading_path": sec.path},
                    start_line=sec.start + ps + 1,
                    end_line=sec.start + pe + 1,
                    part=n if len(parts) > 1 else None,
                    of=len(parts) if len(parts) > 1 else None,
                )
            )
    return units


# ---------------------------------------------------------------- emit
def _clean(meta: dict[str, Any]) -> dict[str, MetadataValue]:
    """Drop what Chroma rejects (None, empty strings, empty lists) and make
    list values sorted + unique so chunks stay byte-identical between runs."""
    out: dict[str, MetadataValue] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float):
            out[key] = value
        elif isinstance(value, str):
            if value != "":
                out[key] = value
        elif isinstance(value, list | tuple | set):
            items = sorted({str(v) for v in value if str(v) != ""})
            if items:
                out[key] = items
    return out


def _emit(source: SourceFile, ctx: _Ctx, units: list[_Unit]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for unit in units:
        parts = (
            [(unit.body, 0, unit.body.count("\n"))]
            if ctx.count(unit.body) <= ctx.budget
            else _window_split(unit.body, ctx.budget, ctx.overlap_tokens, ctx.count)
        )
        for n, (body, ps, pe) in enumerate(parts, start=1):
            if not body.strip():
                continue
            suffix = f"#{n}" if len(parts) > 1 else ""
            header = f"[{ctx.repo_id}] {source.path} | {unit.title}"
            part_no, part_of = (n, len(parts)) if len(parts) > 1 else (unit.part, unit.of)
            if part_no:
                header += f" (part {part_no}/{part_of})"
            text, redactions = redact_secrets(f"{header}\n{body}")

            own = {unit.meta.get("resource_name"), *(unit.meta.get("resource_names") or [])}
            mentions = [m for m in ctx.registry.mentions(text) if m not in own]
            deprecated = bool(re.search(r"\bdeprecated\b", text, re.IGNORECASE)) or any(
                ctx.registry.names.get(m) == "deprecated" for m in mentions
            )
            meta: dict[str, Any] = {
                "repo_id": ctx.repo_id,
                "path": source.path,
                "file_type": source.file_type,
                "chunk_kind": unit.kind,
                "unit": unit.key + suffix,
                **unit.meta,
                "mentions": mentions,
                "mentions_deprecated": deprecated,
                "redacted": redactions > 0,
            }
            if (
                unit.meta.get("resource_name") or unit.meta.get("resource_names")
            ) and not unit.meta.get("status"):
                meta["status"] = "active"
            if unit.start_line is not None:
                meta["start_line"] = unit.start_line + (ps if len(parts) > 1 else 0)
                meta["end_line"] = (
                    unit.start_line + pe if len(parts) > 1 else (unit.end_line or unit.start_line)
                )
            if part_no:
                meta["part"] = part_no

            base_id = f"{ctx.repo_id}::{source.path}::{unit.key}{suffix}"
            count = seen.get(base_id, 0)
            seen[base_id] = count + 1
            chunks.append(
                Chunk(
                    id=base_id if count == 0 else f"{base_id}~{count + 1}",
                    text=text,
                    metadata=_clean(meta),
                )
            )
    return chunks


# ---------------------------------------------------------------- public API
def chunk_file(
    source: SourceFile,
    *,
    repo_id: str,
    registry: Registry | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int = MIN_CHUNK_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk one file. ``registry`` (from the whole repo) enables handler->job
    linking and resource mentions; without it chunks are still valid, just
    less annotated."""
    default_max, default_overlap = (
        _default_sizes() if max_tokens is None or overlap_tokens is None else (0, 0)
    )
    ctx = _Ctx(
        repo_id=repo_id,
        registry=registry or Registry(),
        max_tokens=max_tokens or default_max,
        overlap_tokens=default_overlap if overlap_tokens is None else overlap_tokens,
        min_tokens=min_tokens,
        count=count_tokens or _default_counter(),
    )
    if source.file_type == "py":
        units = _python_units(source, ctx)
    elif source.file_type in ("yaml", "json"):
        units = _structured_file_units(source, ctx)
    elif source.file_type == "md":
        units = _markdown_units(source, ctx)
    else:
        units = _text_units(source, ctx)
    return _emit(source, ctx, units)


def chunk_files(
    sources: list[SourceFile],
    *,
    repo_id: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int = MIN_CHUNK_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk a whole repo: builds the registry first, then chunks every file
    in path order so the output is stable."""
    registry = Registry.from_sources(sources)
    chunks: list[Chunk] = []
    for source in sorted(sources, key=lambda s: s.path):
        chunks.extend(
            chunk_file(
                source,
                repo_id=repo_id,
                registry=registry,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                min_tokens=min_tokens,
                count_tokens=count_tokens,
            )
        )
    return chunks
