"""
Pre-build script for Zensical documentation.

Performs three steps in order:
1. Copy files from outside docs/ into the docs tree (examples, experimental,
   developer docs) — replaces mkdocs-gen-files.
2. Generate API documentation pages with mkdocstrings ::: directives —
   replaces mkdocs-api-autonav.
3. Read docs/.nav.yml, expand glob patterns, and write the TOML nav array
   into zensical.toml — replaces mkdocs-awesome-nav.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# API docs configuration
# ---------------------------------------------------------------------------
SRC_ROOT = Path("src")
TOP_MODULE = "llmcompressor"
API_DOCS_DIR = Path("docs/api")
API_EXCLUDE = {"llmcompressor.version", "llmcompressor.typing"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Find the project root by looking for zensical.toml."""
    current = Path(__file__).resolve().parent
    while True:
        if (current / "zensical.toml").exists():
            return current
        if current == current.parent:  # Reached the root
            break
        current = current.parent
    raise FileNotFoundError("Could not find zensical.toml")

# ---------------------------------------------------------------------------
# Copy files into docs/
# ---------------------------------------------------------------------------


@dataclass
class CopySpec:
    root_path: Path
    docs_path: Path
    title: str | None = None
    weight: float | None = None


def copy_files(specs: list[CopySpec], project_root: Path):
    docs_dir = project_root / "docs"
    for spec in specs:
        source = project_root / spec.root_path
        target = docs_dir / spec.docs_path

        if not source.exists():
            print(f"  Skipping {source} (not found)")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_text(encoding="utf-8")

        if spec.title is not None or spec.weight is not None:
            frontmatter = "---\n"
            if spec.title is not None:
                frontmatter += f"title: {spec.title}\n"
            if spec.weight is not None:
                frontmatter += f"weight: {spec.weight}\n"
            frontmatter += "---\n\n"
            content = frontmatter + content

        target.write_text(content, encoding="utf-8")


def _collect_readmes(src_dir: Path, docs_subdir: str, project_root: Path):
    """Collect README.md files from a directory of sub-projects."""
    specs: list[CopySpec] = []
    if not src_dir.exists():
        return specs

    main_readme = src_dir / "README.md"
    if main_readme.exists():
        specs.append(
            CopySpec(
                root_path=main_readme.relative_to(project_root),
                docs_path=Path(f"{docs_subdir}/README.md"),
            )
        )

    for child in sorted(src_dir.iterdir()):
        readme = child / "README.md"
        if child.is_dir() and readme.exists():
            specs.append(
                CopySpec(
                    root_path=readme.relative_to(project_root),
                    docs_path=Path(f"{docs_subdir}/{child.name}.md"),
                )
            )
    return specs


def generate_copied_files(project_root: Path):
    """Copy examples, experimental, and developer docs into docs/."""
    specs: list[CopySpec] = []

    # Examples and experimental READMEs
    specs.extend(
        _collect_readmes(project_root / "examples", "examples", project_root)
    )
    specs.extend(
        _collect_readmes(
            project_root / "experimental", "experimental", project_root
        )
    )

    copy_files(specs, project_root)


# ---------------------------------------------------------------------------
# Generate API pages
# ---------------------------------------------------------------------------


def _get_module_path(pkg_dir: Path, src_root: Path) -> str:
    return ".".join(pkg_dir.relative_to(src_root).parts)


def _should_exclude(module_path: str) -> bool:
    for excl in API_EXCLUDE:
        if module_path == excl or module_path.startswith(excl + "."):
            return True
    return False


def _write_api_page(doc_file: Path, module_path: str, title: str):
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text(
        f"---\ntitle: {title}\n---\n\n::: {module_path}\n",
        encoding="utf-8",
    )


def generate_api_pages(project_root: Path) -> int:
    """Generate API markdown files. Returns number of modules processed."""
    src_root = project_root / SRC_ROOT
    api_dir = project_root / API_DOCS_DIR
    top_module_dir = src_root / TOP_MODULE

    # Clean previously generated API subdirectories (preserves top-level files)
    for item in api_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    count = 0
    for init_file in sorted(top_module_dir.rglob("__init__.py")):
        pkg_dir = init_file.parent
        pkg_module_path = _get_module_path(pkg_dir, src_root)

        if _should_exclude(pkg_module_path):
            continue

        rel_parts = pkg_dir.relative_to(src_root).parts
        doc_dir = api_dir / Path(*rel_parts)
        _write_api_page(doc_dir / "index.md", pkg_module_path, rel_parts[-1])
        count += 1

        for py_file in sorted(pkg_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            module_name = py_file.stem
            module_path = f"{pkg_module_path}.{module_name}"

            if _should_exclude(module_path):
                continue

            _write_api_page(
                doc_dir / module_name / "index.md", module_path, module_name
            )
            count += 1

    return count


# ---------------------------------------------------------------------------
# Generate TOML nav from .nav.yml
# ---------------------------------------------------------------------------


def _get_title_from_file(file_path: Path) -> str:
    """Extract title from markdown frontmatter or first heading."""
    try:
        content = file_path.read_text(encoding="utf-8")
        # Check YAML frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                frontmatter = content[3:end]
                for line in frontmatter.strip().split("\n"):
                    if line.startswith("title:"):
                        return line[6:].strip().strip('"').strip("'")
        # Check first H1 heading
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip().rstrip("#").strip()
    except Exception:
        pass

    # Derive from filename
    name = file_path.stem
    if name in ("index", "README"):
        name = file_path.parent.name
    return name.replace("-", " ").replace("_", " ").title()


def _expand_glob_dir(
    base_dir: Path,
    docs_dir: Path,
    exclude_paths: set[str] | None = None,
) -> list:
    """Expand a directory into a nested nav structure.

    Subdirectories become sections, markdown files become leaf pages.
    index.md/README.md files are added as bare paths (section indexes).
    Files in exclude_paths are skipped to avoid duplicates.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    if exclude_paths is None:
        exclude_paths = set()

    entries = []
    index_files = []
    regular_files = []
    subdirs = []

    for item in sorted(base_dir.iterdir()):
        if item.is_file() and item.suffix == ".md":
            rel = str(item.relative_to(docs_dir))
            if rel in exclude_paths:
                continue
            if item.name in ("index.md", "README.md"):
                index_files.append(item)
            else:
                regular_files.append(item)
        elif item.is_dir() and not item.name.startswith("."):
            subdirs.append(item)

    for idx_file in index_files:
        entries.append(str(idx_file.relative_to(docs_dir)))

    for subdir in subdirs:
        children = _expand_glob_dir(subdir, docs_dir, exclude_paths)
        if children:
            if (subdir / "index.md").exists():
                title = _get_title_from_file(subdir / "index.md")
            else:
                title = (
                    subdir.name.replace("-", " ").replace("_", " ").title()
                )
            entries.append({title: children})

    for f in regular_files:
        title = _get_title_from_file(f)
        entries.append({title: str(f.relative_to(docs_dir))})

    return entries


def _collect_explicit_paths(items: list) -> set[str]:
    """Collect all explicit (non-glob) file paths from a nav item list."""
    paths = set()
    for item in items:
        if isinstance(item, str) and "*" not in item:
            paths.add(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and "*" not in value:
                    paths.add(value)
    return paths


def _process_nav_item(
    item,
    docs_dir: Path,
    sibling_paths: set[str] | None = None,
) -> list:
    """Process a single YAML nav item. Returns a list of nav entries."""
    if sibling_paths is None:
        sibling_paths = set()

    if isinstance(item, str):
        if "*" in item:
            glob_dir = docs_dir / item.replace("/*", "").replace("*", "")
            return _expand_glob_dir(glob_dir, docs_dir, sibling_paths)
        return [item]

    if isinstance(item, dict):
        results = []
        for title, value in item.items():
            if isinstance(value, str):
                if "*" in value:
                    expanded = _expand_glob_dir(
                        docs_dir / value.replace("/*", "").replace("*", ""),
                        docs_dir,
                        sibling_paths,
                    )
                    results.append({title: expanded})
                else:
                    results.append({title: value})
            elif isinstance(value, list):
                child_explicit = _collect_explicit_paths(value)
                children = []
                for child in value:
                    children.extend(
                        _process_nav_item(child, docs_dir, child_explicit)
                    )
                results.append({title: children})
            elif value is None:
                results.append({title: []})
        return results

    return []


def _escape_toml(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _nav_to_toml(nav_items: list, indent: int = 4) -> str:
    prefix = " " * indent
    lines = []

    for item in nav_items:
        if isinstance(item, str):
            lines.append(f'{prefix}"{_escape_toml(item)}",')
        elif isinstance(item, dict):
            for title, value in item.items():
                t = _escape_toml(title)
                if isinstance(value, str):
                    v = _escape_toml(value)
                    lines.append(f'{prefix}{{ "{t}" = "{v}" }},')
                elif isinstance(value, list):
                    if not value:
                        lines.append(f'{prefix}{{ "{t}" = [] }},')
                    else:
                        lines.append(f'{prefix}{{ "{t}" = [')
                        lines.append(_nav_to_toml(value, indent + 4))
                        lines.append(f"{prefix}]}},")

    return "\n".join(lines)


def _replace_nav_block(config: str, new_nav_block: str) -> str:
    """Replace the nav = [...] block in TOML, handling nested brackets."""
    match = re.search(r"^nav\s*=\s*\[", config, re.MULTILINE)
    if not match:
        raise ValueError("Could not find 'nav = [' in zensical.toml")

    start = match.start()
    bracket_start = match.end() - 1

    depth = 0
    pos = bracket_start
    while pos < len(config):
        ch = config[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = pos + 1
                return config[:start] + new_nav_block + config[end:]
        elif ch == "#":
            while pos < len(config) and config[pos] != "\n":
                pos += 1
        elif ch in ('"', "'"):
            quote = ch
            pos += 1
            while pos < len(config) and config[pos] != quote:
                if config[pos] == "\\":
                    pos += 1
                pos += 1
        pos += 1

    raise ValueError("Could not find closing ] for nav array")


def generate_nav(project_root: Path) -> int:
    """Read .nav.yml and write the TOML nav into zensical.toml.

    Returns the number of top-level nav entries.
    """
    docs_dir = project_root / "docs"
    nav_file = docs_dir / ".nav.yml"
    config_file = project_root / "zensical.toml"

    with open(nav_file, encoding="utf-8") as f:
        nav_yaml = yaml.safe_load(f)

    raw_nav = nav_yaml.get("nav", [])

    top_level_paths = _collect_explicit_paths(raw_nav)
    nav_items = []
    for item in raw_nav:
        nav_items.extend(_process_nav_item(item, docs_dir, top_level_paths))

    toml_nav = _nav_to_toml(nav_items)
    nav_block = f"nav = [\n{toml_nav}\n]"

    config = config_file.read_text(encoding="utf-8")
    config = _replace_nav_block(config, nav_block)
    config_file.write_text(config, encoding="utf-8")

    return len(nav_items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clean_nav(project_root: Path):
    """Reset nav in zensical.toml to an empty array."""
    config_file = project_root / "zensical.toml"
    config = config_file.read_text(encoding="utf-8")
    config = _replace_nav_block(config, "nav = []")
    config_file.write_text(config, encoding="utf-8")


def main():
    import sys

    project_root = find_project_root()

    if "--clean" in sys.argv:
        print("Resetting nav in zensical.toml ...")
        clean_nav(project_root)
        print("Done.")
        return

    print("Copying files into docs/ ...")
    generate_copied_files(project_root)

    print("Generating API pages ...")
    api_count = generate_api_pages(project_root)
    print(f"  {api_count} API modules")

    print("Generating nav in zensical.toml ...")
    nav_count = generate_nav(project_root)
    print(f"  {nav_count} top-level entries")

    print("Done.")


if __name__ == "__main__":
    main()
