# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from pathlib import Path

import pytest

from tools.lint_research import lint_entry, main


@pytest.fixture(scope="module")
def guide_examples():
    guide = Path(__file__).resolve().parents[2] / "docs/research/index.md"
    blocks = re.findall(
        r"^```markdown\n(.*?)^```$", guide.read_text("utf-8"), re.M | re.S
    )
    report = next(block for block in blocks if block.startswith("---\n"))
    byline = next(block for block in blocks if block.startswith("<div "))
    report = report.replace("USERNAME", "alice-test").replace(
        "YYYY-MM-DD", "2026-01-10"
    )
    return report, byline


@pytest.fixture
def entry(tmp_path, guide_examples):
    path = tmp_path / "awq-qwen3.5-alice-test.md"
    path.write_text(guide_examples[0], encoding="utf-8")
    return path


def test_guide_examples(entry, guide_examples):
    assert lint_entry(entry) == []

    report, byline = guide_examples
    byline = byline.strip().replace("YYYY-MM-DD", "2026-01-10**, updated **2026-02-05")
    report = re.sub(r"<div .*?</div>", lambda _: byline, report, count=1, flags=re.S)
    coauthored = entry.with_name("awq-qwen3.5-carol.md")
    coauthored.write_text(report, encoding="utf-8")
    assert lint_entry(coauthored) == []


@pytest.mark.parametrize(
    ("before", "after", "issue"),
    [
        ("---\n", "", "YAML front matter between ---"),
        ("title:", "title: [", "invalid YAML"),
        ("title:", "unused:", "nonempty title"),
        ("title:", "title: Duplicate\ntitle:", "declare 'title' only once"),
        ("tags:", "tags: [AWQ, GPTQ]\nunused:", "block-style tags"),
        ("  - ", "  - # ", "text tag"),
        ('class="entry-byline" markdown', 'class="entry-byline"', "entry-byline block"),
        ("\n# ", "\n# ###\n", "level-one Markdown heading"),
        (".entry-avatar", ".ENTRY-AVATAR", "linked avatar"),
        ("[![alice-test]", "    [![alice-test]", "linked avatar"),
        ("@alice-test", "@other", "linked GitHub handle"),
        ("posted on", "posted", "posted on **YYYY-MM-DD**"),
        ("2026-01-10", "2026-02-30", "valid posted date"),
        ("2026-01-10**", "2026-01-10**, updated **20260205**", "valid updated date"),
    ],
)
def test_rejects_malformed_entry(entry, guide_examples, before, after, issue):
    entry.write_text(guide_examples[0].replace(before, after, 1), encoding="utf-8")
    assert any(issue in message for message in lint_entry(entry))


@pytest.mark.parametrize(
    "filename", ["awq-other.md", "-alice-test.md", "awq-alice-test.txt"]
)
def test_filename_format(entry, filename):
    wrong_name = entry.rename(entry.with_name(filename))
    assert lint_entry(wrong_name) == [
        "name the file <short-title>-<credited-github-handle>.md"
    ]


def test_cli_scans_nested_reports_and_accepts_explicit_paths(
    entry, monkeypatch, capsys
):
    monkeypatch.chdir(entry.parent)
    group = Path("docs/research/sweeps")
    group.mkdir(parents=True)
    for name in ("index.md", "README.md"):
        (group / name).write_text("# Category\n", encoding="utf-8")
    broken = group / "awq-alice.md"
    broken.write_text("# Missing metadata\n", encoding="utf-8")

    assert main([]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"{broken}: start the entry with YAML front matter between --- delimiters"
    ]
    assert main([str(entry)]) == 0
