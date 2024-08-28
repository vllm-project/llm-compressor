import pytest

from types import SimpleNamespace

from llmcompressor.utils import (
    ALL_TOKEN,
    convert_to_bool,
    flatten_iterable,
    interpolate,
    parse_kwarg_tuples,
    validate_str_iterable,
    getattr_chain,
)


@pytest.mark.parametrize(
    "test_list,output",
    [
        ([], []),
        ([0, 1], [0, 1]),
        ([[0, 1], [2, 3]], [0, 1, 2, 3]),
        ([[0, 1], 2, 3], [0, 1, 2, 3]),
    ],
)
def test_flatten_iterable(test_list, output):
    flattened = flatten_iterable(test_list)
    assert flattened == output


@pytest.mark.parametrize(
    "test_bool,output",
    [
        (True, True),
        ("t", True),
        ("T", True),
        ("true)", True),
        ("True", True),
        (1, True),
        ("1", True),
        (False, False),
        ("f", False),
        ("F", False),
        ("false", False),
        ("False", False),
        (0, False),
        ("0", False),
    ],
)
def test_convert_to_bool(test_bool, output):
    converted = convert_to_bool(test_bool)
    assert converted == output


@pytest.mark.parametrize(
    "test_list,output",
    [
        (ALL_TOKEN, ALL_TOKEN),
        (ALL_TOKEN.lower(), ALL_TOKEN),
        ([], []),
        ([0, 1], [0, 1]),
        ([[0], [1]], [0, 1]),
    ],
)
def test_validate_str_iterable(test_list, output):
    validated = validate_str_iterable(test_list, "")
    assert validated == output


def test_validate_str_iterable_negative():
    with pytest.raises(ValueError):
        validate_str_iterable("will fail", "")


@pytest.mark.parametrize(
    "x_cur,x0,x1,y0,y1,inter_func,out",
    [
        (0.0, 0.0, 1.0, 0.0, 5.0, "linear", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "cubic", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 0.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "linear", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "cubic", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 5.0),
        (0.5, 0.0, 1.0, 0.0, 5.0, "linear", 2.5),
        (0.5, 0.0, 1.0, 0.0, 5.0, "cubic", 4.375),
        (0.5, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 1.031),
    ],
)
def test_interpolate(x_cur, x0, x1, y0, y1, inter_func, out):
    interpolated = interpolate(x_cur, x0, x1, y0, y1, inter_func)
    assert abs(out - interpolated) < 0.01


def test_pass_kwargs_tuples():
    kwargs = parse_kwarg_tuples(("--input_1", 1, "--input_2", "two", "--input_3", "2"))
    assert kwargs == dict(input_1=1, input_2="two", input_3=2)


def test_getattr_chain():
    base = SimpleNamespace()
    base.a = None
    base.b = SimpleNamespace()
    base.b.c = "value"
    base.b.d = None

    # test base cases
    assert getattr_chain(base, "", None) is None
    with pytest.raises(AttributeError):
        getattr_chain(base, "")

    # test single layer
    assert getattr_chain(base, "a") is None
    assert getattr_chain(base, "a", "default") is None
    assert getattr_chain(base, "b") == base.b

    assert getattr_chain(base, "dne", None) is None
    with pytest.raises(AttributeError):
        getattr_chain(base, "dne")

    # test multi layer
    assert getattr_chain(base, "b.c") == "value"
    assert getattr_chain(base, "b.d") is None
    assert getattr_chain(base, "b.d", "default") is None
    
    assert getattr_chain(base, "b.d.dne", "default") == "default"
    with pytest.raises(AttributeError):
        getattr_chain(base, "b.d.dne")