"""LUT-B GPTQ with codebooks fitted on first use of each column tile.

Unlike a static fit, this quantizer waits until GPTQ reaches a column tile for
the first time before choosing that tile's E8M0 range and E4M3 codebook. At
that point the fit sees the tile's current weights, including error updates
propagated from previously processed columns. The selected qparams are then
held fixed for the rest of the tile.

Column tiles are the LUT-B storage groups: each is ``block_k`` consecutive
columns in the original weight. A supplied execution order can interleave
those groups (for example, A, B, A, C, ...), while each group's codebook still
matches the canonical LUT-B tile layout.
"""

import torch
from compressed_tensors.quantization import ActivationOrdering, QuantizationArgs
from compressed_tensors.quantization.utils.lut_b import (
    LUT_B_LLOYD_ITERATIONS,
    _decode_lut_b_scales,
    _fit_codebooks,
    _scale_lut_b_tiles,
    is_lut_b_quantization,
)

from llmcompressor.modifiers.gptq.helpers import (
    GPTQ_PRECISION,
    factorize_hessian,
)
from llmcompressor.modifiers.gptq.lutb import GPTQLutBModifier

__all__ = ["GPTQLutBDelayedModifier", "quantize_weight_lut_b_delayed"]


@torch.no_grad()
def quantize_weight_lut_b_delayed(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float | None = 0.01,
    *,
    column_order: torch.Tensor | None = None,
) -> tuple[float, dict[str, torch.Tensor], bool]:
    """Run GPTQ while fitting each LUT-B tile's qparams when first reached.

    ``column_order`` maps processing positions to original input-column
    indices. When omitted, columns are processed in their original order. The
    group ID for a column is its original column-tile index, so an order whose
    tile IDs are ``A, B, A, C, D, A, B, B, E`` defers E's fit until its first
    column is processed.

    The codebook for a newly reached group is fit from all of that group's
    current weights. Earlier GPTQ error updates have already been applied to
    those weights; qparams are not refit as the remaining columns are reached.

    :param module: A two-dimensional linear module to quantize.
    :param quant_args: Canonical LUT-B block-quantization arguments.
    :param hessian: Normalized GPTQ Hessian for the module's input columns.
    :param blocksize: Number of execution-order columns processed per outer
        chunk. Error propagation still updates every later column.
    :param percdamp: Fraction of the mean Hessian diagonal used for damping.
    :param column_order: Optional permutation from execution position to
        original column index.
    :return: Total GPTQ loss, a weight-only qparam dictionary, and whether
        Hessian factorization fell back to RTN.
    """
    if not is_lut_b_quantization(quant_args):
        raise ValueError(
            "quantize_weight_lut_b_delayed requires canonical LUT-B arguments, "
            f"got {quant_args}"
        )
    if module.weight.ndim != 2:
        raise ValueError(
            "Delayed LUT-B GPTQ currently requires a two-dimensional weight, "
            f"got shape {tuple(module.weight.shape)}"
        )
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square two-dimensional tensor")
    if blocksize <= 0:
        raise ValueError(f"blocksize must be positive, got {blocksize}")

    block_n, block_k = quant_args.block_structure
    codebook_size = 1 << quant_args.num_bits
    original_weight = module.weight.detach()
    num_rows, num_columns = original_weight.shape
    if num_rows % block_n or num_columns % block_k:
        raise ValueError(
            "LUT-B GPTQ requires weight dimensions divisible by "
            f"({block_n}, {block_k}), got ({num_rows}, {num_columns})"
        )
    if hessian.shape[0] != num_columns:
        raise ValueError(
            f"hessian width {hessian.shape[0]} does not match weight width "
            f"{num_columns}"
        )

    device = original_weight.device
    if column_order is None:
        order = torch.arange(num_columns, device=device, dtype=torch.long)
    else:
        order = torch.as_tensor(column_order, device=device, dtype=torch.long)
        if order.ndim != 1 or order.numel() != num_columns:
            raise ValueError(
                "column_order must be a one-dimensional permutation with one "
                f"entry per column; got shape {tuple(order.shape)}"
            )
        if not torch.equal(
            torch.sort(order).values,
            torch.arange(num_columns, device=device, dtype=torch.long),
        ):
            raise ValueError("column_order must contain each column exactly once")

    # Build execution-position lists for each canonical column tile. Sort each
    # list by original column so codebook initialization follows storage order,
    # independent of the order in which GPTQ will consume its columns.
    ordered_columns = order.detach().cpu().tolist()
    positions_by_tile: dict[int, list[int]] = {}
    for position, original_column in enumerate(ordered_columns):
        tile_index = original_column // block_k
        positions_by_tile.setdefault(tile_index, []).append(position)
    for positions in positions_by_tile.values():
        positions.sort(key=ordered_columns.__getitem__)

    weight = original_weight.to(dtype=GPTQ_PRECISION).index_select(1, order)
    hessian = hessian.to(device=device, dtype=GPTQ_PRECISION)
    hessian = hessian.index_select(0, order).index_select(1, order)

    used_rtn_fallback = torch.zeros(1, dtype=torch.bool, device=device)
    dampening = 0.01 if percdamp is None else percdamp
    hessian_inverse = factorize_hessian(
        weight.unsqueeze(0),
        hessian.unsqueeze(0),
        dampening,
        used_rtn_fallback,
    ).squeeze(0)

    num_row_tiles = num_rows // block_n
    row_to_tile = torch.arange(num_rows, device=device) // block_n
    quantized = torch.empty_like(weight)
    losses = torch.zeros(num_rows, dtype=GPTQ_PRECISION, device=device)
    fitted_qparams: dict[int, torch.Tensor] = {}

    for chunk_start in range(0, num_columns, blocksize):
        chunk_end = min(chunk_start + blocksize, num_columns)
        weight_block = weight[:, chunk_start:chunk_end].clone()
        quantized_block = torch.empty_like(weight_block)
        errors = torch.zeros_like(weight_block)
        block_losses = torch.zeros_like(weight_block)
        inverse_block = hessian_inverse[
            chunk_start:chunk_end, chunk_start:chunk_end
        ]

        for offset in range(chunk_end - chunk_start):
            position = chunk_start + offset
            original_column = ordered_columns[position]
            tile_index = original_column // block_k

            if tile_index not in fitted_qparams:
                tile_columns = []
                for tile_position in positions_by_tile[tile_index]:
                    if tile_position < chunk_start:
                        raise RuntimeError(
                            "A LUT-B tile was first fitted after one of its "
                            "columns had already been processed"
                        )
                    if tile_position < chunk_end:
                        tile_columns.append(weight_block[:, tile_position - chunk_start])
                    else:
                        tile_column = weight[:, tile_position]
                        if offset:
                            # Block GPTQ delays propagation to columns after
                            # this block. Include errors accumulated earlier in
                            # this block so the first-use fit sees the same
                            # residual-updated weights as columnwise GPTQ.
                            tile_column = tile_column - errors[:, :offset].matmul(
                                hessian_inverse[
                                    chunk_start:position, tile_position
                                ]
                            )
                        tile_columns.append(tile_column)
                current_tile = torch.stack(tile_columns, dim=1)
                # [row tiles, block_n * block_k] in canonical tile order.
                current_tile = (
                    current_tile
                    .reshape(num_row_tiles, block_n, block_k)
                    .reshape(num_row_tiles, block_n * block_k)
                )
                normalized, encoded_scales = _scale_lut_b_tiles(current_tile)
                _, centers = _fit_codebooks(
                    normalized,
                    quant_args,
                    LUT_B_LLOYD_ITERATIONS,
                )
                scale_factors = _decode_lut_b_scales(encoded_scales).reshape(-1)
                centers = centers.to(torch.float32).reshape(
                    num_row_tiles, codebook_size
                )
                fitted_qparams[tile_index] = centers[row_to_tile] * scale_factors[
                    row_to_tile
                ].unsqueeze(1)

            column_centers = fitted_qparams[tile_index]
            value = weight_block[:, offset]
            nearest = (value.unsqueeze(1) - column_centers).abs().argmin(dim=1)
            quantized_column = column_centers.gather(
                1, nearest.unsqueeze(1)
            ).squeeze(1)
            quantized_block[:, offset] = quantized_column

            diagonal = inverse_block[offset, offset]
            error = (value - quantized_column) / diagonal
            errors[:, offset] = error
            block_losses[:, offset] = error.square()
            weight_block[:, offset:] -= error.unsqueeze(1) * inverse_block[
                offset, offset:
            ].unsqueeze(0)

        weight[:, chunk_start:chunk_end] = quantized_block
        quantized[:, chunk_start:chunk_end] = quantized_block
        losses += block_losses.sum(dim=1) / 2
        if chunk_end < num_columns:
            weight[:, chunk_end:] -= errors.matmul(
                hessian_inverse[chunk_start:chunk_end, chunk_end:]
            )

    restored = torch.empty_like(quantized)
    restored[:, order] = quantized
    return (
        losses.sum().item(),
        {"weight": restored.to(dtype=original_weight.dtype)},
        bool(used_rtn_fallback.item()),
    )


class GPTQLutBDelayedModifier(GPTQLutBModifier):
    """LUT-B GPTQ that defers each tile's codebook fit until first use.

    Activation ordering is enabled by default, allowing the first-use order of
    column-tile groups to differ from storage order. Set ``actorder: null`` to
    process columns in their original order.
    """

    actorder: ActivationOrdering | None = ActivationOrdering.STATIC

    def _quantize_weight(self, module, quant_args, hessian):
        column_order = None
        if self.actorder is not None:
            column_order = torch.argsort(
                torch.diagonal(hessian), descending=True
            )

        return quantize_weight_lut_b_delayed(
            module=module,
            quant_args=quant_args,
            hessian=hessian,
            blocksize=self.block_size,
            percdamp=self.dampening_frac,
            column_order=column_order,
        )
