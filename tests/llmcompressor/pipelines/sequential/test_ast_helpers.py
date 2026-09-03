import subprocess
import sys


def test_autowrap_forward_under_cprofile(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        """
import torch

from llmcompressor.pipelines.sequential.ast_helpers import autowrap_forward

activation = torch.relu


class Model(torch.nn.Module):
    def forward(self, value):
        return activation(value)


model = Model()
with autowrap_forward(model, []):
    model(torch.tensor(-1))
"""
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(tmp_path / "profile.out"),
            str(script),
        ],
        check=True,
    )
