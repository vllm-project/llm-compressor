# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os
import sys
from typing import List, NamedTuple


COPYRIGHT_LINES = [
    "Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.",
    "",
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "",
    "   http://www.apache.org/licenses/LICENSE-2.0",
    "",
    "Unless required by applicable law or agreed to in writing,",
    'software distributed under the License is distributed on an "AS IS" BASIS,',
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "See the License for the specific language governing permissions and",
    "limitations under the License.",
]
NO_COPYRIGHT_LINE = "neuralmagic: no copyright"
QUALITY_COMMAND = "quality"
STYLE_COMMAND = "style"


def parse_args():
    """
    Setup and parse command line arguments for using the script
    """
    parser = argparse.ArgumentParser(
        description=(
            "Add Neuralmagic copyright to the beginning of all "
            "files under the given glob patterns. "
            "Currently assumes Python files using '#' as the commenting prefix."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    quality_parser = subparsers.add_parser(
        QUALITY_COMMAND,
        description=(
            "Run check across the files in the given patterns and "
            "fail if any do not have a copyright in them"
        ),
    )
    style_parser = subparsers.add_parser(
        STYLE_COMMAND,
        description=(
            "Add the copyright to any files in the given patterns if it is not present"
        ),
    )

    for sub in [quality_parser, style_parser]:
        sub.add_argument(
            "patterns",
            type=str,
            default=[],
            nargs="+",
            help="the patterns to search through",
        )

    return parser.parse_args()


def quality(patterns: List[str]):
    """
    Run a quality check across all files in the given glob patterns.
    This checks to make sure all matching files have the NM copyright present.
    If any do not, it will list them out and exit with an error.

    :param patterns: The glob file patterns to run quality check on
    """
    check_files = _get_files(patterns)
    error_files = []

    for file in check_files:
        if not _dont_copyright(file) and not _contains_copyright(file):
            print(f"would add copyright to {file}")
            error_files.append(file)

    if error_files:
        sys.exit(
            f"{len(error_files)} would be copyrighted, "
            f"{len(check_files) - len(error_files)} would be left unchanged."
        )
    else:
        print(f"{len(check_files)} files have copyrights")


def style(patterns: List[str]):
    """
    Run a style application across all files in the given glob patterns.
    This checks to make sure all matching files have the NM copyright present.
    If any do not, it will append the copyright to above the file after
    any already contained headers such as shebang lines.

    :param patterns: The glob file patterns to run quality check on
    """
    check_files = _get_files(patterns)
    copyrighted_files = []

    for file in check_files:
        if not _dont_copyright(file) and not _contains_copyright(file):
            _add_copyright(file)
            print(f"copyrighted {file}")
            copyrighted_files.append(file)

    if copyrighted_files:
        print(
            f"{len(copyrighted_files)} file(s) copyrighted, "
            f"{len(check_files) - len(copyrighted_files)} files unchanged"
        )
    else:
        print(f"{len(check_files)} files unchanged")


def _get_files(patterns: List[str]) -> List[str]:
    files = []

    for pattern in patterns:
        for file in glob.glob(pattern, recursive=True):
            files.append(os.path.abspath(os.path.expanduser(file)))

    files.sort()

    return files


def _dont_copyright(file_path: str) -> bool:
    with open(file_path, "r") as file:
        content = file.read()

    try:
        content.index(NO_COPYRIGHT_LINE)

        return True
    except ValueError:
        return False


def _contains_copyright(file_path: str) -> bool:
    with open(file_path, "r") as file:
        content = file.read()

    try:
        for line in COPYRIGHT_LINES:
            content.index(line)

        return True
    except ValueError:
        return False


def _add_copyright(file_path: str):
    file_type = _file_type(file_path)

    if file_type == "unknown":
        raise ValueError(
            f"unsupported file_type given to be copyrighted at {file_path}"
        )

    with open(file_path, "r+") as file:
        lines = file.readlines()
        header_info = _file_header_info(lines, file_type)
        inject_index = 0

        if header_info.end_index > -1:
            # if there is already a header, we want to inject the copyright after it
            # additionally we'll need a new line between the prev header and copyright
            inject_index = header_info.end_index + 1
            lines.insert(inject_index, "\n")
            inject_index += 1

        # add the copyright at the inject index
        file_copyright = _file_copyright(file_type)
        lines.insert(inject_index, file_copyright)

        if not header_info.new_line_after:
            # if there wasn't a new line after the header,
            # add in a new line after to create space between the code and copyright
            inject_index += 1
            lines.insert(inject_index, "\n")

        file.seek(0)
        file.writelines(lines)
        file.truncate()


def _file_copyright(file_type: str) -> str:
    comment_formatting = _code_comment_formatting(file_type)
    lines = []

    if comment_formatting.block_prefix:
        lines.append(comment_formatting.block_prefix)

    for line in COPYRIGHT_LINES:
        lines.append(
            f"{comment_formatting.line_prefix} {line}"
            if comment_formatting.line_prefix
            else line
        )

    if comment_formatting.block_suffix:
        lines.append(comment_formatting.block_suffix)

    # make sure there is a new line after last line of the copyright
    lines.append("")

    return "\n".join(lines)


_HeaderInfo = NamedTuple(
    "HeaderInfo",
    [
        ("start_index", int),
        ("end_index", int),
        ("new_line_before", bool),
        ("new_line_after", bool),
    ],
)


def _file_header_info(lines: List[str], file_type: str) -> _HeaderInfo:
    start_index = -1
    end_index = -1
    new_line_before = False
    new_line_after = False

    comment_formatting = _code_comment_formatting(file_type)
    prefix_found = False
    suffix_found = False

    for index, line in enumerate(lines):
        line = line.strip()

        if not line:
            # empty line, record the state of new lines before and after header
            if not prefix_found:
                new_line_before = True
            elif prefix_found and (suffix_found or not comment_formatting.block_suffix):
                new_line_after = True
        elif (
            comment_formatting.block_prefix
            and line.startswith(comment_formatting.block_prefix)
        ) or (
            not comment_formatting.block_prefix
            and line.startswith(comment_formatting.line_prefix)
        ):
            # start of header
            prefix_found = True
            start_index = index
            end_index = index
            suffix_found = comment_formatting.block_suffix and line.endswith(
                comment_formatting.block_suffix
            )
        elif comment_formatting.block_suffix and line.endswith(
            comment_formatting.block_suffix
        ):
            # end of header
            suffix_found = True
            end_index = index
        elif prefix_found and comment_formatting.block_suffix and not suffix_found:
            # in the middle of the header, searching for the end
            # reset new_line_after in case there was a break in the header
            new_line_after = True
        else:
            # first non header line, break out
            break

    return _HeaderInfo(start_index, end_index, new_line_before, new_line_after)


_CommentFormatting = NamedTuple(
    "CommentFormatting",
    [
        ("line_prefix", str),
        ("block_prefix", str),
        ("block_suffix", str),
    ],
)


def _code_comment_formatting(file_type: str) -> _CommentFormatting:
    if file_type == "python":
        return _CommentFormatting("#", "", "")
    elif file_type == "html" or file_type == "markdown":
        return _CommentFormatting("", "<!--", "-->")
    elif file_type == "css" or file_type == "javascript":
        return _CommentFormatting("", "/*", "*/")
    elif file_type == "restructuredtext":
        return _CommentFormatting("   ", "..", "")

    raise ValueError(f"unsupported file_type given for code prefix suffix: {file_type}")


def _file_type(file_path: str) -> str:
    if file_path.endswith(".py"):
        return "python"
    elif (
        file_path.endswith(".js")
        or file_path.endswith(".jsx")
        or file_path.endswith(".ts")
        or file_path.endswith(".tsx")
        or file_path.endswith(".jss")
    ):
        return "javascript"
    elif file_path.endswith(".html"):
        return "html"
    elif file_path.endswith(".css"):
        return "css"
    elif file_path.endswith(".md"):
        return "markdown"
    elif file_path.endswith(".rst"):
        return "restructuredtext"

    return "unknown"


def main():
    args = parse_args()

    if args.command == QUALITY_COMMAND:
        quality(args.patterns)
    elif args.command == STYLE_COMMAND:
        style(args.patterns)
    else:
        raise ValueError(f"unknown command given: {args.command}")


if __name__ == "__main__":
    main()