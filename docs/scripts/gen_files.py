"""
Copy required files from outside of the docs directory into the docs directory
for the documentation build and site.
Uses mkdocs-gen-files to handle the file generation and compatibility with MkDocs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mkdocs_gen_files


@dataclass
class ProcessFile:
    root_path: Path
    docs_path: Path
    title: Optional[str] = None
    weight: Optional[float] = None


def find_project_root() -> Path:
    start_path = Path(__file__).absolute()
    current_path = start_path.parent

    while current_path:
        if (current_path / "mkdocs.yml").exists():
            return current_path
        current_path = current_path.parent

    raise FileNotFoundError(
        f"Could not find mkdocs.yml in the directory tree starting from {start_path}"
    )


def process_files(files: list[ProcessFile], project_root: Path):
    for file in files:
        source_path = project_root / file.root_path
        target_path = file.docs_path

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file {source_path} does not exist for copying into docs "
                f"directory at {target_path}"
            )

        content = source_path.read_text(encoding="utf-8")

        # Only add frontmatter if title or weight are set
        if file.title is not None or file.weight is not None:
            frontmatter = "---\n"
            if file.title is not None:
                frontmatter += f"title: {file.title}\n"
            if file.weight is not None:
                frontmatter += f"weight: {file.weight}\n"
            frontmatter += "---\n\n"
            content = frontmatter + content

        with mkdocs_gen_files.open(target_path, "w") as file_handle:
            file_handle.write(content)

        mkdocs_gen_files.set_edit_path(target_path, source_path)


def migrate_developer_docs():
    project_root = find_project_root()
    files = [
        ProcessFile(
            root_path=Path("CODE_OF_CONDUCT.md"),
            docs_path=Path("developer/code-of-conduct.md"),
        ),
        ProcessFile(
            root_path=Path("CONTRIBUTING.md"),
            docs_path=Path("developer/contributing.md"),
        ),
    ]
    process_files(files, project_root)


def migrate_examples():
    project_root = find_project_root()
    examples_path = project_root / "examples"
    files = []

    # Add the main examples README.md
    main_readme = examples_path / "README.md"
    if main_readme.exists():
        files.append(
            ProcessFile(
                root_path=main_readme.relative_to(project_root),
                docs_path=Path("examples/README.md"),
            )
        )

    # Find all README.md files 2 levels down (examples/EXAMPLE_NAME/README.md)
    for example_dir in examples_path.iterdir():
        if (
            not example_dir.is_dir()
            or not (readme_path := example_dir / "README.md").exists()
        ):
            continue

        example_name = example_dir.name
        files.append(
            ProcessFile(
                root_path=readme_path.relative_to(project_root),
                docs_path=Path(f"examples/{example_name}.md"),
            )
        )

    process_files(files, project_root)


def migrate_experimental():
    project_root = find_project_root()
    experimental_path = project_root / "experimental"
    files = []

    # Add the main experimental README.md
    main_readme = experimental_path / "README.md"
    if main_readme.exists():
        files.append(
            ProcessFile(
                root_path=main_readme.relative_to(project_root),
                docs_path=Path("experimental/README.md"),
            )
        )

    # Find all README.md files 2 levels down (experimental/EXPERIMENTAL_NAME/README.md)
    for experimental_dir in experimental_path.iterdir():
        if (
            not experimental_dir.is_dir()
            or not (readme_path := experimental_dir / "README.md").exists()
        ):
            continue

        experimental_name = experimental_dir.name
        files.append(
            ProcessFile(
                root_path=readme_path.relative_to(project_root),
                docs_path=Path(f"experimental/{experimental_name}.md"),
            )
        )

    process_files(files, project_root)


def migrate_readme_to_index():
    """Copy README.md files to index.md for MkDocs compatibility.

    Skips docs/README.md
    """
    project_root = find_project_root()
    docs_path = project_root / "docs"

    for readme_path in docs_path.rglob("README.md"):
        # Skip the root docs/README.md
        if readme_path.parent == docs_path:
            continue

        relative_path = readme_path.relative_to(docs_path)
        target_path = relative_path.parent / "index.md"

        content = readme_path.read_text(encoding="utf-8")

        with mkdocs_gen_files.open(target_path, "w") as file_handle:
            file_handle.write(content)

        mkdocs_gen_files.set_edit_path(target_path, readme_path)


migrate_developer_docs()
migrate_examples()
migrate_experimental()
migrate_readme_to_index()
