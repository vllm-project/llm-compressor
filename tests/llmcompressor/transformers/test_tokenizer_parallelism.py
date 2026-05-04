import os

import pytest

from llmcompressor.entrypoints.oneshot import (
    TOKENIZERS_PARALLELISM_ENV as _TOKENIZERS_PARALLELISM_ENV,
)


class TestTokenizerParallelism:
    """Tests for tokenizer parallelism warning suppression (issue #2007)."""

    def test_oneshot_sets_tokenizers_parallelism_when_not_set(self, monkeypatch):
        """
        Test that Oneshot sets TOKENIZERS_PARALLELISM=false when not already set.

        This prevents the warning:
        "huggingface/tokenizers: The current process just got forked, after
        parallelism has already been used. Disabling parallelism to avoid deadlocks..."

        See: https://github.com/vllm-project/llm-compressor/issues/2007
        """
        monkeypatch.delenv(_TOKENIZERS_PARALLELISM_ENV, raising=False)

        from llmcompressor.entrypoints.oneshot import Oneshot

        # Create a minimal Oneshot instance to trigger __init__
        # We expect it to fail due to missing model, but the env var should be set
        with pytest.raises(Exception):
            Oneshot(model="nonexistent-model")

        assert os.environ[_TOKENIZERS_PARALLELISM_ENV] == "false"

    def test_oneshot_respects_existing_tokenizers_parallelism(self, monkeypatch):
        """
        Test that Oneshot respects user's existing TOKENIZERS_PARALLELISM setting.

        If a user has explicitly set TOKENIZERS_PARALLELISM, we should not override it.
        """
        monkeypatch.setenv(_TOKENIZERS_PARALLELISM_ENV, "true")

        from llmcompressor.entrypoints.oneshot import Oneshot

        with pytest.raises(Exception):
            Oneshot(model="nonexistent-model")

        assert os.environ[_TOKENIZERS_PARALLELISM_ENV] == "true"
