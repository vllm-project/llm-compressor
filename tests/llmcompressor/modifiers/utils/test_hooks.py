import torch

from llmcompressor.modifiers.utils.hooks import HooksMixin


class DummyModel(torch.nn.Module):
    """Dummy Model for testing hooks"""

    def __init__(self):
        super(DummyModel, self).__init__()

        self.linear1 = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(2, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.dummy_inputs = torch.tensor([0.0])

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x


class DummyMod(HooksMixin):
    hook_called: bool = False

    def hook(self, *args, **kwargs):
        self.hook_called = True


class ModA(DummyMod):
    pass


class ModB(DummyMod):
    pass


def test_register_hook():
    model = DummyModel()

    mod_a = ModA()
    mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")

    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called


def test_remove_hooks():
    model = DummyModel()

    mod_a = ModA()
    mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")
    mod_b.remove_hooks()

    model(model.dummy_inputs)
    assert mod_a.hook_called and not mod_b.hook_called


def test_remove_hooks_parameterized():
    model = DummyModel()

    mod_a = ModA()
    mod_a_pre_hook = mod_a.register_hook(model.linear1, mod_a.hook, "forward_pre")
    mod_a_post_hook = mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    mod_b_pre_hook = mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")
    mod_b_post_hook = mod_b.register_hook(model.linear2, mod_b.hook, "forward")

    mod_a.remove_hooks(set([mod_a_post_hook]))
    mod_b.remove_hooks(set([mod_b_pre_hook]))

    assert len(mod_a._hooks) == 1 and next(iter(mod_a._hooks)) == mod_a_pre_hook
    assert len(mod_b._hooks) == 1 and next(iter(mod_b._hooks)) == mod_b_post_hook

    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called


def test_disable_hooks():
    model = DummyModel()

    mod_a = ModA()
    mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")

    with HooksMixin.disable_hooks():
        model(model.dummy_inputs)
    assert not mod_a.hook_called and not mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called


def test_disable_hooks_keep():
    model = DummyModel()

    mod_a = ModA()
    handle_a = mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    handle_b = mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")

    with HooksMixin.disable_hooks(keep=set([handle_b])):
        model(model.dummy_inputs)
    assert not mod_a.hook_called and mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    with HooksMixin.disable_hooks(keep=set([handle_a])):
        model(model.dummy_inputs)
    assert mod_a.hook_called and not mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called


def test_disable_hooks_composable():
    model = DummyModel()

    mod_a = ModA()
    handle_a = mod_a.register_hook(model.linear1, mod_a.hook, "forward")

    mod_b = ModB()
    handle_b = mod_b.register_hook(model.linear2, mod_b.hook, "forward_pre")

    # composing two keeps
    with HooksMixin.disable_hooks(keep=set([handle_b])), HooksMixin.disable_hooks(
        keep=set([handle_a])
    ):
        model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    with HooksMixin.disable_hooks():
        model(model.dummy_inputs)
    assert not mod_a.hook_called and not mod_b.hook_called

    # composing a keep and an empty keep
    mod_a.hook_called = False
    mod_b.hook_called = False
    with HooksMixin.disable_hooks(keep=set([handle_a])), HooksMixin.disable_hooks():
        model(model.dummy_inputs)
    assert mod_a.hook_called and not mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    model(model.dummy_inputs)
    assert mod_a.hook_called and mod_b.hook_called

    mod_a.hook_called = False
    mod_b.hook_called = False
    with HooksMixin.disable_hooks():
        model(model.dummy_inputs)
    assert not mod_a.hook_called and not mod_b.hook_called
