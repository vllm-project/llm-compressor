# ModelSlim ↔ LLMC MXFP differential

**Current result: NOT numerically aligned.** This is a strict parity probe, separate
from the successful transform/lifecycle/export regression gate.

```bash
python tools/run_mxfp_differential.py \
  --modelslim-source /path/to/msmodelslim \
  --report-dir /tmp/mxfp-differential
```

Exit 0 means every compared value matches; **exit 1 means differences were found**.
The JSON reports `parity_passed=false` for the current pinned sources. There are no
real checkpoints, external datasets, GPU/NPU requirements or downloads.

The external source is ModelSlim `beb917d011bc1d11a8bf5e46f62928f6d84b06a7`.
CT is `0.19.1a20260919`, torch `2.14.0+cpu`. Original source definitions are compiled
unchanged: QStorage/QDType, block reshape, minmax observer, weight quantizer driver,
MX scale/Q/DQ functions and W4A4/W8A8 dynamic linear forwards. The adapter replaces
registration and selects these exact functions, supplies standalone state objects
and runs single-process CPU code. It does not run the full processor, API backend
selection, NPU kernels or checkpoint savers. No source algorithm is vendored.

LLMC uses its actual MinMaxObserver for weights, CT dynamic qparams for activations,
CT quantize/dequantize, and QuantizationModifier's real lifecycle for linear output.
Group size is 32 along the last dimension; both sides receive identical FP32/BF16
inputs. Reports include signed exponents, decoded scales, normalized E8M0 bytes,
quantized values, dequantized values and linear outputs, with source hashes.
E8M0 here is numerical encoding comparison, not an Ascend checkpoint layout claim.

## Results, 2026-09-23

104 tensor comparisons: 2 formats × 2 dtypes × 13 cases × 2 roles.
**52 cases differ** in at least one intermediate/output. These deliberately include
boundaries; the fraction is not a model-accuracy estimate. Three of four random
linear output cases differ. The FP32 MXFP4 random linear control matches exactly.

| Minimal case | ModelSlim | LLMC / CT | Cause |
|---|---|---|---|
| MXFP4, scale=1, input=0.25 | 0.5 | 0 | Midpoint rounding: away from zero vs ties-to-even |
| MXFP8, scale=1, input=1.0625 | 1.125 | 1.0 | Same rounding difference |
| MXFP8, block max=7.5 | scale=2^-6, max dequant=7 | scale=2^-5, max dequant=7.5 | Different shared-exponent selection |
| MXFP4 all-zero block | scale=2^-22 | scale=2^-127 | Reference epsilon policy; outputs are both zero |
| MXFP4 input=1e-8 | dequant=0 | dequant≈1.1175871e-8 | Small-value scale policy |
| MXFP4 BF16 block max=6.96875 | scale=2 | scale=1 | Native-BF16 scale arithmetic boundary |

The reference's MXFP4 scale uses max/(7/8) + 9.6e-7 before log2; its MXFP8 scale uses
floor(log2(max)) minus the element exponent offset. CT uses its bit-based scale
rounding routine for both formats. Both reference quantizers use magnitude rounding
via floor(value+0.5), while CT's element casts use ties-to-even. These are existing
implementation conventions; this change introduces no alternative quantizer or
accuracy optimization and does not decide which convention performs better on GLM.

The fixed seed-42 linear fixture uses weights [16,64] and inputs [2,3,64]:

| Format / dtype | Different output elements / 96 | Maximum absolute error |
|---|---:|---:|
| MXFP4 FP32 | 0 | 0 |
| MXFP4 BF16 | 60 | 1.0 |
| MXFP8 FP32 | 16 | 0.28125 |
| MXFP8 BF16 | 89 | 0.625 |

## Harness validation

```bash
python -m pytest tests/mxfp -q -p no:cacheprovider
```

27 checks pass: exact representable controls, positive/negative halfway counterexamples,
scale/epsilon/BF16 cases, per-block noncontiguous 3D inputs, and each real linear
driver matching its own independently assembled Q/DQ components. They validate the
comparison and characterize known differences; **passing these tests does not turn
the strict parity result into a pass**. Missing optional source skips normal pytest;
set `MXFP_REQUIRE_REFERENCE=1` to require it. The standalone probe always requires it.

Next implementation work should define an explicit ModelSlim-compatible scale/rounding
policy and cover both static weights and dynamic activations. Changing only saved
weight scales or only the observer cannot resolve dynamic activation rounding.
Do not silently change standard MXFP preset semantics, patch installed dependencies,
or attribute these differences to QuaRot/FlexSmooth. Mixed-MXFP numerical L2 remains
open; kernel agreement and real L4/L5 are also unverified.
