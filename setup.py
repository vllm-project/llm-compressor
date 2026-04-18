import os
import re
from pathlib import Path

from packaging.version import Version
from setuptools import find_packages, setup
from setuptools_git_versioning import count_since, get_branch, get_sha, get_tags

LAST_RELEASE_VERSION = Version("0.10.0")
TAG_VERSION_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")


def get_last_version_diff() -> tuple[Version, str | None, int | None]:
    """
    Get the last version, last tag, and the number of commits since the last tag.
    If no tags are found, return the last release version and None for the tag/commits.

    :returns: A tuple containing the last version, last tag, and number of commits since
        the last tag.
    """
    tagged_versions = [
        (Version(match.group(1)), tag)
        for tag in get_tags(root=Path(__file__).parent)
        if (match := TAG_VERSION_PATTERN.match(tag))
    ]
    tagged_versions.sort(key=lambda tv: tv[0])
    last_version, last_tag = (
        tagged_versions[-1] if tagged_versions else (LAST_RELEASE_VERSION, None)
    )
    commits_since_last = (
        count_since(last_tag + "^{commit}", root=Path(__file__).parent)
        if last_tag
        else None
    )

    return last_version, last_tag, commits_since_last


def get_next_version(
    build_type: str, build_iteration: str | int | None
) -> tuple[Version, str | None, int]:
    """
    Get the next version based on the build type and iteration.
    - build_type == release: take the last version and add a post if build iteration
    - build_type == candidate: increment to next minor, add 'rc' with build iteration
    - build_type == nightly: increment to next minor, add 'a' with build iteration
    - build_type == alpha: increment to next minor, add 'a' with build iteration
    - build_type == dev: increment to next minor, add 'dev' with build iteration

    :param build_type: The type of build (release, candidate, nightly, alpha, dev).
    :param build_iteration: The build iteration number. If None, defaults to the number
        of commits since the last tag or 0 if no commits since the last tag.
    :returns: A tuple containing the next version, the last tag the version is based
        off of (if any), and the final build iteration used.
    """
    version, tag, commits_since_last = get_last_version_diff()

    if not build_iteration and build_iteration != 0:
        build_iteration = commits_since_last or 0
    elif isinstance(build_iteration, str):
        build_iteration = int(build_iteration)

    if build_type == "release":
        if commits_since_last:
            # add post since we have commits since last tag
            version = Version(f"{version.base_version}.post{build_iteration}")
        return version, tag, build_iteration

    # not in release pathway, so need to increment to target next release version
    version = Version(f"{version.major}.{version.minor + 1}.0")

    if build_type == "candidate":
        # add 'rc' since we are in candidate pathway
        version = Version(f"{version}.rc{build_iteration}")
    elif build_type in ["nightly", "alpha"]:
        # add 'a' since we are in nightly or alpha pathway
        version = Version(f"{version}.a{build_iteration}")
    else:
        # assume 'dev' if not in any of the above pathways
        version = Version(f"{version}.dev{build_iteration}")

    return version, tag, build_iteration


def write_version_files() -> tuple[Path, Path]:
    """
    Write the version information to version.txt and version.py files.
    version.txt contains the version string.
    version.py contains the version plus additional metadata.

    :returns: A tuple containing the paths to the version.txt and version.py files.
    """
    build_type = os.getenv("BUILD_TYPE", "dev").lower()
    version, tag, build_iteration = get_next_version(
        build_type=build_type,
        build_iteration=os.getenv("BUILD_ITERATION"),
    )
    module_path = Path(__file__).parent / "src" / "llmcompressor"
    version_txt_path = module_path / "version.txt"
    version_py_path = module_path / "version.py"

    with version_txt_path.open("w") as file:
        file.write(str(version))

    with version_py_path.open("w") as file:
        file.writelines(
            [
                f'version = "{version}"\n',
                f'build_type = "{build_type}"\n',
                f'build_iteration = "{build_iteration}"\n',
                f'git_commit = "{get_sha()}"\n',
                f'git_branch = "{get_branch()}"\n',
                f'git_last_tag = "{tag}"\n',
            ]
        )

    return version_txt_path, version_py_path


setup(
    setuptools_git_versioning={
        "enabled": True,
        "version_file": str(write_version_files()[0]),
    },
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["llmcompressor", "llmcompressor.*"], exclude=["*.__pycache__.*"]
    ),
)
