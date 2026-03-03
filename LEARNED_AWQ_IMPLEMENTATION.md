# Learned AWQ Scales Implementation

## Overview

Implemented gradient-based learning for AWQ scale optimization as an alternative to the existing grid search method. This provides a more efficient approach to finding optimal per-channel scales for activation-aware weight quantization.

## Changes Made

### 1. Configuration Parameters (`src/llmcompressor/modifiers/awq/base.py`)

Added three new configuration parameters to `AWQModifier`:

```python
scale_search_method: Literal["grid", "learned"] = "grid"  # Method selection
learned_scales_iters: int = 100  # Optimization iterations
learned_scales_lr: float = 0.01  # Learning rate
```

**Backward Compatibility:** Default is `"grid"` to maintain existing behavior.

### 2. Core Implementation

#### Scale Parameter Management

- `_get_scale_param_name(smooth_name)`: Generates unique parameter names using MD5 hash
- `_attach_learnable_scales(mapping, initial_scales)`: Registers scales as `nn.Parameter` on parent module
- `_detach_learnable_scales(mapping)`: Removes parameter and returns final values

#### Forward Hooks for Scale Application

- `_create_scale_application_hook(mapping, orig_weights, smooth_layer_orig_weight, smooth_layer_orig_bias)`: Creates forward pre-hook that applies:
  - **Scales** to balance layer weights: `W_balance * s`
  - **Inverse scales** to smooth layer weights and bias: `W_smooth / s`, `bias_smooth / s`
- Hooks registered on **both** balance layers and smooth layer
- Scales remain attached to computation graph for gradient flow

#### Gradient-Compatible Loss Computation

Modified `_compute_loss` to support both scalar and tensor returns:
- `return_scalar=True`: Returns Python float (existing behavior, for grid search)
- `return_scalar=False`: Returns `torch.Tensor` maintaining computation graph (for learned method)

#### Learned Scale Optimization

New method `_compute_best_scale_learned`:

1. **Initialization:** Computes initial scales from activation statistics (same as grid search ratio=0.5)
2. **Optimization Loop:**
   - Attaches scales as learnable `nn.Parameter`
   - Registers forward hooks on **both balance and smooth layers**
   - Applies scales via hooks during forward pass:
     - Balance layers: multiply weights by `s`
     - Smooth layer: divide weights and bias by `s`
   - Quantizes balance layer weights: `Q(W_balance * s)`
   - Applies inverse scales after quantization: `Q(W * s) / s`
   - Computes MSE loss maintaining gradients
   - Backpropagates through scales
   - Updates scales with Adam optimizer
   - Clamps scales to valid range `[1e-4, inf)` and removes NaN/inf
3. **Cleanup:**
   - Removes hooks from both balance and smooth layers
   - Restores original weights to both balance and smooth layers
   - Detaches and returns best scales

### 3. Integration

Modified `_apply_smoothing` to conditionally call either method based on `scale_search_method`:

```python
if self.scale_search_method == "learned":
    best_scales = self._compute_best_scale_learned(...)
else:
    best_scales = self._compute_best_scale(...)
```

Both methods return the same tensor format, so the rest of the smoothing pipeline remains unchanged.

### 4. Metrics Logging

Updated `_log_error_metrics` to include method-specific configuration:
- Grid method logs: `n_grid`
- Learned method logs: `learned_scales_iters`, `learned_scales_lr`

## Architecture

### Dual-Mode Design

```
┌─────────────────────────────────────┐
│   scale_search_method: "grid" |     │
│                        "learned"    │
└────────────┬────────────────────────┘
             │
   ┌─────────┴──────────┐
   │                    │
   ▼                    ▼
┌──────────┐     ┌────────────────┐
│   Grid   │     │    Learned     │
│  Search  │     │  (Gradient)    │
└────┬─────┘     └────────┬───────┘
     │                    │
     └────────┬───────────┘
              │
              ▼
      ┌──────────────┐
      │ Best Scales  │
      │  (Tensor)    │
      └──────────────┘
```

### Gradient Flow with Hooks

```
┌──────────────────────────────────────────┐
│  Scales (nn.Parameter, requires_grad)    │
└────────────┬─────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌────────────┐  ┌─────────────┐
│ Hook on    │  │ Hook on     │
│ Smooth     │  │ Balance     │
│ Layer      │  │ Layers      │
│ (W/s, b/s) │  │ (W*s)       │
└────┬───────┘  └──────┬──────┘
     │                 │
     └────────┬────────┘
              ▼
   ┌──────────────────────┐
   │  Smooth Layer Out    │
   │  X' = X / s          │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Balance Layer In    │
   │  W*s applied         │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Quantize: Q(W*s)    │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Apply s^-1 to out   │
   │  Q(W*s)/s            │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Forward Pass        │
   │  (X'/s) * (Q(W*s)/s) │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  MSE Loss            │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Backward (∇loss)    │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Update Scales       │
   │  (Adam optimizer)    │
   └──────────────────────┘
```

## Testing

Added comprehensive unit tests in `tests/llmcompressor/modifiers/awq/test_base.py`:

1. **test_learned_scales_config**: Verify configuration parameters
2. **test_scale_parameter_attachment_detachment**: Test parameter lifecycle
3. **test_compute_loss_gradient_flow**: Verify gradient maintenance
4. **test_learned_scales_initialization**: Test initialization from activation statistics
5. **test_scale_hook_applies_correctly**: Verify hook applies scales to balance layers and inverse scales to smooth layer
6. **test_learned_scales_with_duo_scaling**: Test with duo_scaling enabled

All existing AWQ tests pass (22/22), confirming backward compatibility.

## Usage Example

```yaml
AWQModifier:
  scale_search_method: learned  # Use gradient-based learning
  learned_scales_iters: 100     # Number of optimization iterations
  learned_scales_lr: 0.01       # Learning rate for Adam
  duo_scaling: true             # Use both activations and weights
  config_groups:
    group_0:
      targets: ["Linear"]
      weights:
        num_bits: 4
        type: int
        symmetric: false
        strategy: group
        group_size: 128
```

## Benefits

1. **Computational Efficiency**: Continuous optimization vs discrete grid search
2. **Gradient-Based**: Leverages backpropagation for scale refinement
3. **Flexibility**: Hyperparameters (lr, iters) can be tuned per model
4. **Backward Compatible**: Grid search remains default
5. **Same Output Format**: Seamless integration with existing pipeline

## Implementation Notes

- **Memory:** Learned approach requires gradient storage but uses hooks to avoid weight duplication
- **Initialization:** Uses same warm start as grid search (activation statistics with ratio=0.5)
- **Observer Strategy:** Uses `memoryless_minmax` observer (same as grid search)
- **Error Handling:** Maintains same assertions and validation as grid search
- **Smooth Layer Scaling:** During optimization, inverse scales (`1/s`) are applied to smooth layer weights/bias to ensure correct forward pass when the smooth layer is part of the parent module's computation (e.g., v_proj → o_proj mappings). This maintains the mathematical identity: `(W_smooth/s) * X * (W_balance*s) ≈ W_smooth * X * W_balance`

## Future Enhancements

Potential improvements for future iterations:
- Learning rate scheduling
- Early stopping based on convergence
- Adaptive hyperparameter tuning
- Multi-stage optimization (coarse-to-fine)
