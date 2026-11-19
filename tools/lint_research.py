# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check Open Research Hub entries against docs/research/index.md."""

import argparse
import re
from datetime import date
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

RESEARCH_DIR = Path("docs/research")
FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)^---[ \t]*(?:\n|$)", re.M | re.S)
BYLINE = re.compile(
    r'(?:[ \t]*\n)*<div class="entry-byline" markdown>[ \t]*\n'
    r"(.*?)^</div>[ \t]*(?:\n|$)",
    re.M | re.S,
)
HANDLE = r"(?P<handle>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)"
PROFILE = r"(?ai:https://github\.com/(?P=handle))"
AVATAR = re.compile(
    r" {0,3}"
    rf"\[!\[{HANDLE}\]\({PROFILE}\.png\?size=64\)"
    rf"\{{[ \t]*\.entry-avatar[ \t]*\}}\]\({PROFILE}\)"
)
CREDIT = re.compile(
    r"(?P<authors>.+) posted on \*\*(?P<posted>[^*]*)\*\*"
    r"(?:, updated \*\*(?P<updated>[^*]*)\*\*)?"
)


def _is_text(node: yaml.Node | None) -> bool:
    return (
        isinstance(node, ScalarNode)
        and node.tag == "tag:yaml.org,2002:str"
        and bool(node.value.strip())
    )


def _lint_front_matter(source: str) -> list[str]:
    try:
        metadata = yaml.compose(source, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        return [f"invalid YAML front matter: {error}"]
    if not isinstance(metadata, MappingNode) or metadata.tag != "tag:yaml.org,2002:map":
        return ["front matter must be a YAML mapping with title and tags"]

    issues = []
    fields = {}
    for key, value in metadata.value:
        if _is_text(key) and key.value in ("title", "tags"):
            if key.value in fields:
                issues.append(f"declare {key.value!r} only once in the front matter")
            fields[key.value] = value

    if not _is_text(fields.get("title")):
        issues.append("add a nonempty title to the front matter")
    tags = fields.get("tags")
    if not (
        isinstance(tags, SequenceNode)
        and tags.tag == "tag:yaml.org,2002:seq"
        and tags.flow_style is False
        and tags.value
        and all(_is_text(tag) for tag in tags.value)
    ):
        issues.append("use a nonempty block-style tags list with one text tag per item")
    return issues


def _is_iso_date(value: str) -> bool:
    try:
        # fromisoformat also accepts compact dates and ISO week dates.
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _lint_byline(path: Path, source: str) -> list[str]:
    lines = source.strip("\n").splitlines()
    handles = []
    for line in lines:
        if not line.lstrip().startswith("[!"):
            break
        avatar = AVATAR.fullmatch(line.rstrip())
        if avatar is None:
            return ["use the template's linked avatar with the .entry-avatar class"]
        handles.append(avatar["handle"].lower())

    sentence = "\n".join(lines[len(handles) :])
    credit = CREDIT.fullmatch(" ".join(sentence.split()))
    if credit is None:
        return [
            "use 'posted on **YYYY-MM-DD**' and optionally 'updated **YYYY-MM-DD**'"
        ]

    issues = []
    links = re.split(r",? and |, ", credit["authors"].lower())
    expected = [f"**[@{name}](https://github.com/{name})**" for name in handles]
    stem = path.stem.lower()
    if links != expected or len(set(handles)) != len(handles):
        issues.append("pair each linked GitHub handle with one avatar in credit order")
    elif (
        path.suffix != ".md"
        or not re.fullmatch(r"\w\S*", stem)
        or not any(stem.endswith(f"-{name}") for name in handles)
    ):
        issues.append("name the file <short-title>-<credited-github-handle>.md")

    for field in ("posted", "updated"):
        value = credit[field]
        if value is not None and not _is_iso_date(value):
            issues.append(f"use a valid {field} date in YYYY-MM-DD format")
    return issues


def lint_entry(path: Path) -> list[str]:
    if path.name in ("index.md", "README.md"):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [f"could not read entry as UTF-8: {error}"]

    front_matter = FRONT_MATTER.match(source)
    if front_matter is None:
        return ["start the entry with YAML front matter between --- delimiters"]
    issues = _lint_front_matter(front_matter[1])
    byline = BYLINE.match(source, front_matter.end())
    if byline is None:
        return issues + [
            "put the hub template's entry-byline block after the front matter"
        ]
    heading = re.match(r"(?:[ \t]*\n)* {0,3}#[ \t]+([^\n]*)", source[byline.end() :])
    if heading is None or not heading[1].strip().strip("#"):
        issues.append("add a nonempty level-one Markdown heading after the byline")
    return issues + _lint_byline(path, byline[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown entries to check")
    args = parser.parse_args(argv)
    if not args.paths and not RESEARCH_DIR.is_dir():
        parser.error(f"{RESEARCH_DIR} was not found. Run from the repository root.")

    failed = False
    for path in args.paths or sorted(RESEARCH_DIR.rglob("*.md")):
        for issue in lint_entry(path):
            print(f"{path}: {issue}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
