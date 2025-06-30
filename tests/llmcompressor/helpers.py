# flake8: noqa


def valid_recipe_strings():
    return [
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
                MagnitudePruningModifier:
                    start: 5
                    end: 10
                    init_sparsity: 0.1
                    final_sparsity: 0.5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test1_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test1_stage:
            constant_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                    targets: __ALL_PRUNABLE__
            magnitude_modifiers:
                MagnitudePruningModifier:
                    start: 5
                    end: 10
                    init_sparsity: 0.1
                    final_sparsity: 0.5
                    targets: __ALL_PRUNABLE__
        """,
        """
        test1_stage:
            smoothquant_modifiers:
                SmoothQuantModifier:
                    smoothing_strength: 0.5
                    mappings: [
                        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
                    ]
        """,
    ]
