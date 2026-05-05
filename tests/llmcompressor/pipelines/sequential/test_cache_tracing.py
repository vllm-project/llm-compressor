import torch
from transformers.cache_utils import DynamicCache

from llmcompressor.pipelines.sequential.transformers_helpers import HFTracer


class DummyModelWithCache(torch.nn.Module):
    """
    Generates the following table after tracing and `graph.print_tabular()`:

    opcode         name              target                                           args
    -------------  ----------------  -----------------------------------------------  ------------------------------------
    placeholder    x                 x                                                ()
    call_function  dynamic_cache     <class 'transformers.cache_utils.DynamicCache'>  ()
    call_module    linear            linear                                           (x,)
    call_module    linear_1          linear                                           (x,)
    call_method    update            update                                           (dynamic_cache, linear, linear_1, 0)
    call_function  getitem           <built-in function getitem>                      (update, 0)
    call_function  getitem_1         <built-in function getitem>                      (update, 1)
    call_method    get_seq_length    get_seq_length                                   (dynamic_cache, 0)
    call_function  add               <built-in function add>                          (getitem, get_seq_length)
    call_function  getattr_1         <built-in function getattr>                      (dynamic_cache, 'layers')
    call_function  getitem_2         <built-in function getitem>                      (getattr_1, 0)
    call_method    get_seq_length_1  get_seq_length                                   (getitem_2,)
    call_function  add_1             <built-in function add>                          (getitem_1, get_seq_length_1)
    output         output            output                                           ((add, add_1, dynamic_cache),)
    """  # noqa: E501

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        cache = DynamicCache()
        key_states = self.linear(x)
        value_states = self.linear(x)

        if torch.fx._symbolic_trace._is_fx_tracing_flag:
            # `DynamicCache` has been monkeypatched to ProxyableDynamicCache
            # but still betrays itself via the `__name__` attribute
            assert DynamicCache.__name__ == "ProxyableDynamicCache"

            # `cache` has its `__class__` attribute overwritten to look like
            # `DynamicCache`, but betrays itself by implementing the below method
            assert hasattr(cache, "install_orig_cache_cls")

        return self._dummy_cache_ops(key_states, value_states, cache)

    def _dummy_cache_ops(self, key_states, value_states, cache: DynamicCache):
        key_states, value_states = cache.update(key_states, value_states, 0)
        key_states += cache.get_seq_length(0)
        value_states += cache.layers[0].get_seq_length()

        return key_states, value_states, cache


def test_dynamic_cache_produces_hf_cache_proxy_node():
    model = DummyModelWithCache()
    tracer = HFTracer()
    graph = tracer.trace(model, dummy_inputs={"x": torch.randn(1, 10)})

    nodes = {node.name: node for node in graph.nodes}

    # DynamicCache is traced as a call_function node
    assert "dynamic_cache" in nodes
    cache_node = nodes["dynamic_cache"]
    assert cache_node.op == "call_function"
    assert cache_node.target is DynamicCache

    # cache.update() is traced as a call_method on the cache proxy
    assert "update" in nodes
    update_node = nodes["update"]
    assert update_node.op == "call_method"
    assert update_node.args[0] is cache_node

    # cache.get_seq_length() is traced as a call_method on the cache proxy
    assert "get_seq_length" in nodes
    assert nodes["get_seq_length"].op == "call_method"
    assert nodes["get_seq_length"].args[0] is cache_node

    # the cache proxy is passed through to the output
    output_node = nodes["output"]
    assert cache_node in output_node.args[0]
