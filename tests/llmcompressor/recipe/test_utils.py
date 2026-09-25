from llmcompressor.recipe.utils import append_recipe_dict


def test_append_recipe_dict_numbers_duplicate_stages():
    first = {"test_stage": {"first": {}}}
    second = {"test_stage": {"second": {}}}

    result = append_recipe_dict(first, second)

    assert result == {
        "test_stage_0": {"first": {}},
        "test_stage_1": {"second": {}},
    }


def test_append_recipe_dict_preserves_existing_numbered_stage():
    first = {
        "test_stage": {"unnumbered": {}},
        "test_stage_0": {"numbered": {}},
    }
    second = {"test_stage": {"incoming": {}}}

    result = append_recipe_dict(first, second)

    assert result == {
        "test_stage_1": {"unnumbered": {}},
        "test_stage_0": {"numbered": {}},
        "test_stage_2": {"incoming": {}},
    }
    assert list(result) == ["test_stage_1", "test_stage_0", "test_stage_2"]
