import os
from dataclasses import dataclass
from typing import Optional


@dataclass()
class VersionInfo:
    version_base: str
    build_type: str
    version: str
    version_major: int
    version_minor: int
    version_patch: int
    version_build: Optional[str]

    @property
    def is_release(self) -> bool:
        return self.build_type == "release"

    @property
    def is_nightly(self) -> bool:
        return self.build_type == "nightly"

    @property
    def is_dev(self) -> bool:
        return self.build_type.startswith("dev")


def extract_version_info(package_path: str) -> VersionInfo:
    """
    Load version and release info from the package
    """
    print(f"Extracting version info from {package_path}")
    version_path = os.path.join(package_path, "version.py")

    # exec() cannot set local variables so need to manually
    locals_dict = {}
    exec(open(version_path).read(), globals(), locals_dict)
    print(locals_dict)
    version_base = locals_dict.get("version_base", "unknown")
    build_type = locals_dict.get("build_type", "unknown")
    version = locals_dict.get("version", "unknown")
    version_major = locals_dict.get("version_major", -1)
    version_minor = locals_dict.get("version_minor", -1)
    version_patch = locals_dict.get("version_patch", -1)
    version_build = locals_dict.get("version_build", None)

    print(f"Loaded version {version} from {version_path}")

    return VersionInfo(
        version_base=version_base,
        build_type=build_type,
        version=version,
        version_major=version_major,
        version_minor=version_minor,
        version_patch=version_patch,
        version_build=version_build,
    )
