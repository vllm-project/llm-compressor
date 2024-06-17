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


import os
from typing import Tuple


def get_release_and_version(package_path: str) -> Tuple[bool, bool, str, str, str, str]:
    """
    Load version and release info from deepsparse package
    """
    # deepsparse/src/deepsparse/version.py always exists, default source of truth
    version_path = os.path.join(package_path, "version.py")

    # exec() cannot set local variables so need to manually
    locals_dict = {}
    exec(open(version_path).read(), globals(), locals_dict)
    is_release = locals_dict.get("is_release", False)
    version = locals_dict.get("version", "unknown")
    version_major = locals_dict.get("version_major", "unknown")
    version_minor = locals_dict.get("version_minor", "unknown")
    version_bug = locals_dict.get("version_bug", "unknown")

    print(f"Loaded version {version} from {version_path}")

    return (
        is_release,
        version,
        version_major,
        version_minor,
        version_bug,
    )
