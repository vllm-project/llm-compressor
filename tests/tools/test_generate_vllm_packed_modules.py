import json
import os
import subprocess
import sys

import pytest

from tools.generate_vllm_packed_modules import (
    build_snapshot,
    collect_file,
    collect_modules,
    main,
    read_model_files,
)

PACKED = "packed_modules_mapping"
STACKED = "stacked_params_mapping"


PACKED_DICT = """
class Model:
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ("gate_proj", "up_proj"),
        "skipped": ["a_proj", name],
        "also_skipped": [],
        **other,
    }
"""
PACKED_OTHER_FORMS = """
STEP5_PACKED_MODULES_MAPPING: dict = {"qkv_proj": ["q_proj", "k_proj"]}

class Merged(Base):
    packed_modules_mapping = Base.packed_modules_mapping | {"ba": ["b", "a"]}

class Model:
    def __init__(self):
        self.packed_modules_mapping["in_proj"] = ["in_proj_x", "in_proj_z"]
        self.packed_modules_mapping.update({"fc": ["fc1", "fc2"]})
        self.packed_modules_mapping.pop({"ignored": ["a", "b"]})
        self.other_mapping["ignored"] = ["a", "b"]
        packed_modules_mapping: dict
"""
STACKED_TUPLES = """
def load_weights(self, weights):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("gate_up_proj", "up_proj", 1),
        ("not", "a", "pair", "tuple"),
        ("bad", name, 0),
    ]
"""
STACKED_OTHER_FORMS = """
def load_weights(self, weights):
    mla_params_mapping = (("fused_qkv_a_proj", "q_a_proj", 0),) + [
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1)
    ]
    expert_params_mapping = make_expert_params_mapping()
    params = [("ignored", "a", 0)]
    load(orig_to_new_stacked={"w1": ("w13", 0), "w3": ("w13", 1), "x": ("y",)})
    load(orig_to_new_prefix={"a": ("b", 0)})
    weights_mapping: list

def other(self):
    stacked_params_mapping: list = [("in_proj", "in_proj_qkvz", 0)]
    stacked_params_mapping += [("in_proj", "in_proj_ba", 1)]
"""
# tuples appended under a condition join the group of their function body
STACKED_MERGED_PER_BODY = """
class Model:
    def load_weights(self, weights):
        stacked_params_mapping = [("qkv_proj", "q_proj", "q")]
        if self.config.use_k:
            stacked_params_mapping.append(("qkv_proj", "k_proj", "k"))
            stacked_params_mapping.extend([("qkv_proj", "v_proj", "v")])
        stacked_params_mapping.insert(0, ("qkv_proj", "x_proj", "x"))

    def load_other(self, weights):
        stacked_params_mapping = [("qkv_proj", "g_proj", "g")]
"""
# a {packed: (sub, sub)} dict must not read as {sub: (packed, shard)}
PACKED_PAIR_VALUES = """
packed_modules_mapping = {"gate_up_proj": ("gate_proj", "up_proj")}
"""


@pytest.mark.parametrize(
    "source, packed, stacked",
    [
        pytest.param(
            PACKED_DICT,
            [
                ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
                ("gate_up_proj", ["gate_proj", "up_proj"]),
            ],
            [],
            id="packed-dict",
        ),
        pytest.param(
            PACKED_OTHER_FORMS,
            [
                ("qkv_proj", ["q_proj", "k_proj"]),
                ("ba", ["b", "a"]),
                ("in_proj", ["in_proj_x", "in_proj_z"]),
                ("fc", ["fc1", "fc2"]),
            ],
            [],
            id="packed-other-forms",
        ),
        pytest.param(
            STACKED_TUPLES,
            [],
            [
                ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
                ("gate_up_proj", ["gate_proj", "up_proj"]),
            ],
            id="stacked-tuples",
        ),
        pytest.param(
            STACKED_OTHER_FORMS,
            [],
            [
                ("fused_qkv_a_proj", ["q_a_proj", "kv_a_proj_with_mqa"]),
                ("w13", ["w1", "w3"]),
                ("in_proj", ["in_proj_qkvz", "in_proj_ba"]),
            ],
            id="stacked-other-forms",
        ),
        pytest.param(
            STACKED_MERGED_PER_BODY,
            [],
            [
                ("qkv_proj", ["q_proj", "k_proj", "v_proj", "x_proj"]),
                ("qkv_proj", ["g_proj"]),
            ],
            id="stacked-merged-per-body",
        ),
        pytest.param(
            PACKED_PAIR_VALUES,
            [("gate_up_proj", ["gate_proj", "up_proj"])],
            [],
            id="packed-pair-values",
        ),
    ],
)
def test_collect_file(source, packed, stacked):
    assert collect_file(source) == {PACKED: packed, STACKED: stacked}


MODULE_LAYERS = """
class CrossAttention(nn.Module):
    def __init__(self):
        self.q_proj = ColumnParallelLinear(1, 1)
        self.kv_proj = MergedColumnParallelLinear(1, [1, 1])
        self.g_proj = ReplicatedLinear(1, 1) if gated else None
        self.out_proj: nn.Module = nn.Linear(1, 1)
        self.norm = RMSNorm(1)
        self.attn: Attention
        self.attn_proj = make_layer(1, 1)
        other.k_proj = ColumnParallelLinear(1, 1)

    class Gate(nn.Module):
        def __init__(self):
            self.o_proj = RowParallelLinear(1, 1)

class MLP(nn.Module):
    def __init__(self):
        self.down_proj = RowParallelLinear(1, 1)
        self.experts = nn.ModuleList([RowParallelLinear(1, 1) for _ in range(2)])

class Model(nn.Module):
    packed_modules_mapping = {"kv_proj": ["k_proj", "v_proj"]}
"""


def test_collect_modules():
    """
    The linear layers each class assigns to self, not those of a nested class
    or of a container of linear layers
    """
    assert collect_modules(MODULE_LAYERS) == [
        ("CrossAttention", ["g_proj", "kv_proj", "out_proj", "q_proj"]),
        ("MLP", ["down_proj"]),
        ("Gate", ["o_proj"]),
    ]


def test_snapshot_keeps_the_modules_that_pack_layers():
    snapshot = build_snapshot("abc123", {"m.py": MODULE_LAYERS})
    assert snapshot["modules"] == [
        {
            "file": "m.py",
            "class": "CrossAttention",
            "layers": ["g_proj", "kv_proj", "out_proj", "q_proj"],
        }
    ]


def test_snapshot_strips_the_common_parent_and_merges_groups():
    sources = {
        "b.py": 'packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj"]}',
        "a.py": """
stacked_params_mapping = [
    ("model.attn.qkv_proj.weight", "model.attn.k_proj.weight", "k"),
    ("model.attn.qkv_proj.weight", "model.attn.q_proj.weight", "q"),
]
packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj"]}
""",
        "c.py": 'packed_modules_mapping = {"a.qkv.w.x": ["a.q.v.x"]}',
    }
    assert build_snapshot("abc123", sources) == {
        "vllm_commit": "abc123",
        "groups": [
            {
                "packed": "qkv.w",
                "names": ["q.v"],
                "kinds": [PACKED],
                "files": ["c.py"],
            },
            {
                "packed": "qkv_proj",
                "names": ["q_proj", "k_proj"],
                "kinds": [PACKED, STACKED],
                "files": ["a.py", "b.py"],
            },
        ],
        "modules": [],
    }


def test_identity_group():
    # vLLM declares a layer it stores fused as its own only sub
    source = 'packed_modules_mapping = {"qkv_proj": ["qkv_proj"]}'
    snapshot = build_snapshot("abc123", {"m.py": source})
    assert snapshot["groups"] == [
        {
            "packed": "qkv_proj",
            "names": ["qkv_proj"],
            "kinds": [PACKED],
            "files": ["m.py"],
        }
    ]


@pytest.fixture(autouse=True)
def _no_git_environment(monkeypatch):
    # git hooks and `git rebase --exec` set GIT_DIR, which would point the
    # test repositories' commands at the calling repository
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo(tmp_path, files):
    repo = tmp_path / "vllm"
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "models")
    return repo


def test_reads_model_files_from_git_not_the_worktree(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "vllm/models/llama.py": "x = 1\n",
            "vllm/models/README.md": "not python\n",
            "vllm/other.py": "y = 1\n",
        },
    )
    (repo / "vllm/models/llama.py").write_text("x = 2\n")
    (repo / "vllm/models/untracked.py").write_text("z = 1\n")

    commit, sources = read_model_files(repo, "HEAD")

    assert len(commit) == 40
    assert sources == {"vllm/models/llama.py": "x = 1\n"}


def test_missing_model_directories_stop_with_an_error(tmp_path):
    repo = _repo(tmp_path, {"vllm/other.py": "y = 1\n"})

    with pytest.raises(SystemExit, match="none of vllm/model_executor/models"):
        read_model_files(repo, "HEAD")


def test_main_writes_the_snapshot(tmp_path, monkeypatch, capsys):
    source = 'packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj"]}\n'
    repo = _repo(tmp_path, {"vllm/model_executor/models/llama.py": source})
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["gen", str(repo), "--output", str(output)])

    main()

    assert capsys.readouterr().out == f"wrote 1 groups and 0 modules to {output}\n"
    text = output.read_text()
    assert text.endswith("}\n")
    assert json.loads(text)["groups"] == [
        {
            "packed": "qkv_proj",
            "names": ["q_proj", "k_proj"],
            "kinds": [PACKED],
            "files": ["vllm/model_executor/models/llama.py"],
        }
    ]


def test_help_shows_the_summary(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gen", "--help"])

    with pytest.raises(SystemExit):
        main()

    help_text = capsys.readouterr().out
    assert "Collect the linear-layer groups that vLLM packs" in help_text
    assert "Usage:" not in help_text
