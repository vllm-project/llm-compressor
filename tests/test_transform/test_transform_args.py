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

from compressed_tensors.transform import TransformArgs


def test_basic():
    targets = ["Embedding"]
    location = "input"
    args = TransformArgs(targets=targets, location=location)

    assert args.targets == targets
    assert args.location == location
    assert len(args.ignore) == 0


def test_args_full():
    targets = ["Linear"]
    location = "weight_input"
    inverse = True
    ignore = ["model.layers.2"]

    args = TransformArgs(
        targets=targets,
        location=location,
        inverse=inverse,
        ignore=ignore,
    )

    args.targets = targets
    args.location == location
    args.inverse == inverse
    args.ignore == ignore


def test_singleton_targets():
    target = "target"
    location = "input"
    ignore = "ignore"
    args = TransformArgs(targets=target, location=location, ignore=ignore)

    assert args.targets == [target]
    assert args.location == location
    assert args.ignore == [ignore]
