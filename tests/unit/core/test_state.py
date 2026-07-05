import pytest

from llmcompressor.core.state import Data, Hardware, ModifiedState, State


@pytest.mark.smoke
def test_state_initialization():
    state = State()
    assert state.model is None
    assert state.teacher_model is None
    assert state.optimizer is None
    assert state.optim_wrapped is None
    assert state.loss is None
    assert state.batch_data is None
    assert state.data == Data()
    assert state.hardware == Hardware()


@pytest.mark.smoke
def test_modified_state_initialization():
    mod_state = ModifiedState(
        model="model",
        optimizer="optimizer",
        loss="loss",
        modifier_data=[{"key": "value"}],
    )
    assert mod_state.model == "model"
    assert mod_state.optimizer == "optimizer"
    assert mod_state.loss == "loss"
    assert mod_state.modifier_data == [{"key": "value"}]


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
