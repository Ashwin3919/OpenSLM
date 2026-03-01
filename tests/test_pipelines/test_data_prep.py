"""Tests for DataPrepPipeline tokenisation logic (no network calls)."""

import pytest

from src.models.config import AppConfig
from src.pipelines.data_prep import DataPrepPipeline


@pytest.fixture
def pipeline_with_enc(tmp_path):
    """DataPrepPipeline with encoding pre-loaded (skips HuggingFace download)."""
    import tiktoken

    config = AppConfig()
    config.data.output_dir = str(tmp_path / "data")
    pipeline = DataPrepPipeline(config)
    pipeline.configure()
    return pipeline


def test_tokenize_returns_expected_keys(pipeline_with_enc):
    result = pipeline_with_enc._tokenize({"text": "Hello world"})
    assert "ids" in result
    assert "len" in result


def test_tokenize_length_matches(pipeline_with_enc):
    result = pipeline_with_enc._tokenize({"text": "The quick brown fox"})
    assert len(result["ids"]) == result["len"]


def test_tokenize_produces_ints(pipeline_with_enc):
    result = pipeline_with_enc._tokenize({"text": "Once upon a time"})
    assert all(isinstance(t, int) for t in result["ids"])


def test_tokenize_empty_string(pipeline_with_enc):
    result = pipeline_with_enc._tokenize({"text": ""})
    assert result["ids"] == []
    assert result["len"] == 0
