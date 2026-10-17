# GLM calibrated oneshot: sequential boundaries

This is a locally verified integration setting, not an eight-device L5 launch
script. The existing `mixed_mxfp.yaml` policy and Modifier order are unchanged.

```python
oneshot(
    model=model,
    processor=processor,
    dataset=calibration_loader,
    recipe=recipe,
    pipeline="sequential",
    sequential_targets=[
        r"re:.*\.input_layernorm$",
        "ExpertMLP",
        "GlmMoeDsaMLP",
    ],
    sequential_targets_per_subgraph=1,
    propagate_error=True,
    moe_calibrate_all_experts=False,
)
```

The input norm starts the attention partition. This keeps both norm-linear
FlexSmooth groups and the V/O group inside one calibration boundary. Dense and
shared MLPs and individual routed experts form separate partitions. Targeting
`GlmMoeDsaAttention` instead of the input norm still fails the intentional
FlexSmooth split guard. Increasing targets per subgraph trades memory for fewer
passes and has not been profiled here.

Two tracing fixes are needed in this branch:

- A chained method with starred arguments, such as
  `self.experts(...).view(*orig_shape)`, must leave the receiver call visible to FX.
  Wrapping the entire expression hides the expert partition targets.
- Expert accumulation must explicitly depend on each preceding result. In-place
  `index_add_` has no such FX data edge and can execute after a consumer or lose
  its update across activation-cache copies. Functional `index_add` preserves
  the dependency; it also allocates a new accumulator, whose server cost remains
  to be measured.

The official two-layer tiny GLM produces seven actual partitions: a head, two
norm/attention groups, dense/shared MLPs and two routed experts. Tests inspect
module membership and replay every partition with copied inputs, checking exact
final hidden states against an unsplit forward in FP32 and BF16. Copies model
the loss of storage aliasing during CPU/accelerator transfer; they do not test
NPU kernels or device memory peaks. Full QuaRot → FlexSmooth → mixed MXFP oneshot,
post-transform observers, and real compressed save/reload also pass.

`moe_calibrate_all_experts=False` is appropriate to test for this recipe because
FlexSmooth collects attention inputs and MXFP activations use dynamic scales.
A forced-empty expert regression verifies that its static weight scales and
export are still complete. Every expert is still called, including zero-token
experts; this does not establish proportional weight-I/O savings. Do not reuse
the setting for other recipes needing expert activation statistics without
separate validation.

The uncapped distributed Modifier path now passes tiny two-rank sequential tests
with real shared CPU/disk backing and compressed reload. Global token caps remain
deferred, and NPU/full-model profiling is a server task. QuaRot
still runs its full weight traversal at calibration start; these partition
settings do not make rotation layer-streamed. Use explicit QuaRot FP32 for the
future server comparison to the FP32 L4 reference, and record BF16 storage
rounding separately. MTP preservation/export also remains a server prerequisite.

See the [collective search](distributed_search.md) for source comparisons and the
[oneshot profile driver](oneshot_profile.md) for the integrated execution path.
