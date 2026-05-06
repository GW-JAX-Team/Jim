#!/usr/bin/env python3
"""Generate API reference stubs and produce _zensical_build.toml.

Run before: uv run zensical build -f _zensical_build.toml --clean
For local development: uv run zensical serve -f _zensical_build.toml

Environment variables:
    VERSION_ALIAS  When set (e.g. "stable", "latest", "development"),
                   appends the alias to site_url in the output config so
                   that versioned builds have correct canonical URLs.
"""

import os
import re
import shutil
import tomllib
from pathlib import Path

# Module path prefixes (relative to src/) to exclude from the API docs.
# Use forward slashes, e.g. "mypkg/internal".
SKIP_PREFIXES: list[str] = []


# ── module scanning ───────────────────────────────────────────────────────────


def scan_modules(src_dir: Path) -> list[tuple[list[str], Path]]:
    """Return (module_parts, stub_path) for every public Python module."""
    results = []
    for py_file in sorted(src_dir.rglob("*.py")):
        parts = list(py_file.relative_to(src_dir).with_suffix("").parts)
        if parts[-1] in ("__init__", "__main__"):
            continue
        if any(part.startswith("_") for part in parts):
            continue
        rel = "/".join(parts)
        if any(rel == skip or rel.startswith(skip + "/") for skip in SKIP_PREFIXES):
            continue
        stub_path = Path("api", *parts).with_suffix(".md")
        results.append((parts, stub_path))
    return results


def write_stubs(modules: list[tuple[list[str], Path]], docs_dir: Path) -> None:
    """Write (or overwrite) ::: identifier stubs under docs/api/."""
    api_dir = docs_dir / "api"
    keep = {docs_dir / stub_path for _, stub_path in modules}
    if api_dir.exists():
        for existing in api_dir.rglob("*.md"):
            if existing not in keep:
                existing.unlink()
    for parts, stub_path in modules:
        full_path = docs_dir / stub_path
        # Remove any conflicting stale entry left from a previous run
        assert api_dir in full_path.parents, f"stub path escapes api_dir: {full_path}"
        if full_path.exists() and full_path.is_dir():
            shutil.rmtree(full_path)
        if full_path.parent.exists() and full_path.parent.is_file():
            full_path.parent.unlink()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"::: {'.'.join(parts)}\n", encoding="utf-8")
    # Prune empty directories left behind by removed stubs
    if api_dir.exists():
        for d in sorted(api_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and d != api_dir and not any(d.iterdir()):
                d.rmdir()


# ── nav generation ────────────────────────────────────────────────────────────


def build_nav_tree(modules: list[tuple[list[str], Path]]) -> list:
    """Convert flat module list into a nested nav structure."""
    tree: dict = {}
    for parts, stub_path in modules:
        node = tree
        for part in parts[:-1]:
            existing = node.get(part)
            if isinstance(existing, str):
                node[part] = {"__init__": existing}
            node = node.setdefault(part, {})
        leaf = stub_path.as_posix()
        existing = node.get(parts[-1])
        if isinstance(existing, dict):
            existing["__init__"] = leaf
        else:
            node[parts[-1]] = leaf

    def dict_to_nav(d: dict) -> list:
        result = []
        for k, v in d.items():
            if isinstance(v, dict):
                children = []
                if "__init__" in v:
                    children.append({"Overview": v["__init__"]})
                children.extend(
                    dict_to_nav({ck: cv for ck, cv in v.items() if ck != "__init__"})
                )
                result.append({k: children})
            else:
                result.append({k: v})
        return result

    return dict_to_nav(tree)


def _toml_escape(s: str) -> str:
    """Escape backslashes and double-quotes for TOML basic strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def nav_to_toml_str(nav: list, depth: int = 0) -> str:
    """Serialize a nav list to the TOML array-of-inline-tables format."""
    inner = "    " * (depth + 1)
    lines = ["["]
    for item in nav:
        if isinstance(item, str):
            # Bare path entry (no label), e.g. "tutorials/index.md"
            lines.append(f'{inner}"{_toml_escape(item)}",')
        else:
            for k, v in item.items():
                if isinstance(v, str):
                    lines.append(
                        f'{inner}{{"{_toml_escape(k)}" = "{_toml_escape(v)}"}},'
                    )
                else:
                    nested = nav_to_toml_str(v, depth + 1)
                    lines.append(f'{inner}{{"{_toml_escape(k)}" = {nested}}},')
    lines.append("    " * depth + "]")
    return "\n".join(lines)


# ── TOML patching ─────────────────────────────────────────────────────────────


def replace_nav(toml_text: str, new_nav: list) -> str:
    """Replace the nav = [...] block in TOML text with a new nav list."""
    lines = toml_text.splitlines(keepends=True)
    start = end = None
    depth = 0
    for i, line in enumerate(lines):
        # NOTE: This regex only strips double-quoted strings on a single line.
        # It does not handle single-quoted strings, triple-quoted strings, or
        # backslash-escaped quotes. In practice zensical.toml uses only
        # double-quoted single-line strings in the nav block, so this is
        # sufficient, but a full TOML parser would be more robust.
        stripped = re.sub(r'"[^"]*"', "", line)
        if start is None and re.match(r"^\s*nav\s*=\s*\[", line):
            start = i
            depth = stripped.count("[") - stripped.count("]")
            if depth == 0:
                end = i
                break
        elif start is not None:
            depth += stripped.count("[") - stripped.count("]")
            if depth <= 0:
                end = i
                break
    if start is None:
        raise ValueError("nav key not found in zensical.toml")
    if end is None:
        raise ValueError("unterminated nav array in zensical.toml")
    new_block = "nav = " + nav_to_toml_str(new_nav) + "\n"
    return "".join([*lines[:start], new_block, *lines[end + 1 :]])


def patch_site_url(toml_text: str, new_url: str) -> str:
    """Replace site_url value in TOML text."""
    escaped_url = _toml_escape(new_url)
    new_text, count = re.subn(
        r'^(\s*site_url\s*=\s*")([^"]*)(")',
        lambda m: f"{m.group(1)}{escaped_url}{m.group(3)}",
        toml_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError(
            "site_url not found in zensical.toml; cannot patch versioned URL"
        )
    return new_text


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    src_dir = repo_root / "src"
    toml_path = repo_root / "zensical.toml"
    out_path = repo_root / "_zensical_build.toml"

    toml_text = toml_path.read_text(encoding="utf-8")
    config = tomllib.loads(toml_text)

    # Copy CONTRIBUTING.md so Zensical picks it up (symlinks and the name
    # "contributing.md" are both excluded by Zensical at build time)
    shutil.copy(repo_root / "CONTRIBUTING.md", docs_dir / "contributing.md")

    # Generate stubs and build API nav
    modules = scan_modules(src_dir)
    write_stubs(modules, docs_dir)
    api_nav = build_nav_tree(modules)
    print(f"Generated {len(modules)} API stubs")

    # Rebuild nav: keep everything except existing API entry, append new one
    def drop_api_entries(nav):
        """Return a copy of *nav* with any dict that has an "API" key removed."""
        return [item for item in nav if not (isinstance(item, dict) and "API" in item)]

    base_nav = drop_api_entries(config.get("project", {}).get("nav", []))
    new_nav = [*base_nav, {"API": api_nav}]

    new_toml = replace_nav(toml_text, new_nav)

    # Optionally set versioned site_url
    alias = os.environ.get("VERSION_ALIAS", "").strip()
    if alias:
        site_url = config.get("project", {}).get("site_url")
        if site_url:
            base_url = site_url.rstrip("/") + "/"
            new_toml = patch_site_url(new_toml, f"{base_url}{alias}/")
        else:
            print(
                "Warning: VERSION_ALIAS set but site_url not found in zensical.toml; skipping URL patch"
            )

    out_path.write_text(new_toml, encoding="utf-8")
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
