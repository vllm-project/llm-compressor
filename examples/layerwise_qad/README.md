# QAD

The modifier is now named `QADModifier`. See the [QAD example](../qad/README.md)
for RTN/GPTQ initialization, automatic teacher targets, and joint subgraph training.

The old `LayerwiseQADModifier` import and recipe name remain compatibility aliases.
The `distill_teacher` argument is no longer needed or accepted by `oneshot`.
