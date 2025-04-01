import dataclasses
from typing import Any, NewType

from transformers import HfArgumentParser

DataClassType = NewType("DataClassType", Any)


class ArgumentParser(HfArgumentParser):
    def _add_dataclass_arguments(self, dtype: DataClassType):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        # do not collect type hints

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            self._parse_dataclass_field(parser, field)
