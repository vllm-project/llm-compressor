import pytest

from llmcompressor.transformers.compression.helpers import (
    _reduce_targets_and_ignores_into_lists,
)


@pytest.mark.parametrize(
    "exhaustive_targets, exhaustive_ignore, expected_targets, expected_ignore",
    [
        # Test case 1: when exhaustive_targets has more elements
        # than exhaustive_ignore for a given key
        (
            {"type1": ["target1", "target2"], "type2": ["target3"]},
            {"type1": ["ignore1"], "type2": ["ignore2", "ignore3"]},
            ["type1", "target3"],
            ["ignore1", "ignore2", "ignore3"],
        ),
        # Test case 2: when exhaustive_ignore has more elements
        # than exhaustive_targets for a given key
        (
            {"type1": ["target1"], "type2": ["target3", "target4"]},
            {"type1": ["ignore1", "ignore2"], "type2": ["ignore3"]},
            ["target1", "type2"],
            ["ignore1", "ignore2", "ignore3"],
        ),
        # Test case 3: when exhaustive_targets and exhaustive_ignore
        # have the same number of elements for a given key
        (
            {"type1": ["target1", "target2"], "type2": ["target3", "target4"]},
            {"type1": ["ignore1", "ignore2"], "type2": ["ignore3", "ignore4"]},
            ["type1", "type2"],
            ["ignore1", "ignore2", "ignore3", "ignore4"],
        ),
        # Test case 4: when exhaustive_targets
        # and exhaustive_ignore are empty
        ({}, {}, [], []),
        # Test case 5: when exhaustive_targets has
        # keys not present in exhaustive_ignore
        (
            {"type1": ["target1", "target2"], "type3": ["target3", "target4"]},
            {"type1": ["ignore1", "ignore2"], "type2": ["ignore3", "ignore4"]},
            ["type1", "type3"],
            ["ignore1", "ignore2"],
        ),
        # Test case 6: when exhaustive_ignore has keys
        # not present in exhaustive_targets
        (
            {"type1": ["target1", "target2"], "type2": ["target3", "target4"]},
            {"type1": ["ignore1", "ignore2"], "type3": ["ignore3", "ignore4"]},
            ["type1", "type2"],
            ["ignore1", "ignore2"],
        ),
    ],
)
def test_reduce_targets_and_ignores_into_lists(
    exhaustive_targets, exhaustive_ignore, expected_targets, expected_ignore
):
    targets, ignore = _reduce_targets_and_ignores_into_lists(
        exhaustive_targets, exhaustive_ignore
    )
    _assert_list_contents_match(targets, expected_targets)
    _assert_list_contents_match(ignore, expected_ignore)


def _assert_list_contents_match(list1, list2):
    assert len(list1) == len(list2)
    for item in list1:
        assert item in list2
