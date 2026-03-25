# Developer Guides

These guides are for contributors who want to extend LLM Compressor with new functionality. Each guide walks through the relevant abstractions, the contracts you must fulfill, and a concrete working example.

## Tutorials

<div class="grid cards" markdown>

- **Adding a New Modifier**

    ---

    Learn how the modifier lifecycle works and how to implement a custom modifier that integrates with `oneshot`.

    [:octicons-arrow-right-24: Adding a New Modifier](add-modifier.md)

- **Adding a New Observer**

    ---

    Learn how observers compute quantization parameters and how to implement a custom observer registered for use in recipes.

    [:octicons-arrow-right-24: Adding a New Observer](add-observer.md)

- **Adding MoE Calibration Support for a New Model**

    ---

    Learn why MoE models require special calibration handling and how to implement a calibration-friendly module definition for a new MoE architecture.

    [:octicons-arrow-right-24: Adding MoE Calibration Support for a New Model](add-moe-support.md)

</div>
