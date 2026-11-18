import pytest

from llmcompressor.core.state import State


@pytest.mark.smoke
def test_state_update():
    state = State()
    updated_data = {
        "model": "new_model",
        "teacher_model": "new_teacher_model",
        "optimizer": "new_optimizer",
        "train_data": "new_train_data",
        "val_data": "new_val_data",
        "test_data": "new_test_data",
        "calib_data": "new_calib_data",
        "device": "cpu",
        "start": 1.0,
        "batches_per_step": 10,
    }
    state.update(**updated_data)

    assert state.model == "new_model"
    assert state.teacher_model == "new_teacher_model"
    assert state.optimizer == "new_optimizer"
    assert state.data.train == "new_train_data"
    assert state.data.val == "new_val_data"
    assert state.data.test == "new_test_data"
    assert state.data.calib == "new_calib_data"
    assert state.hardware.device == "cpu"


@pytest.mark.regression
def test_state_sparsification_ready():
    state = State()
    assert not state.compression_ready

    state.model = "model"
    state.optimizer = "optimizer"
    assert state.compression_ready
