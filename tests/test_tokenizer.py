"""Token-counter selection and the local tokenizer.json path."""

import pytest

from intelligent_chunker.tokenizer import (
    HeuristicTokenCounter,
    HFTokenCounter,
    get_token_counter,
)


def _write_tiny_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    tok.pre_tokenizer = Whitespace()
    tok.save(str(path))


def test_loads_local_tokenizer_file(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tiny_tokenizer(path)
    counter = HFTokenCounter(str(path))
    assert counter.count("hello world") == 2


def test_missing_local_file_falls_back_to_heuristic(tmp_path, caplog):
    pytest.importorskip("tokenizers")
    # A path-looking id that doesn't exist is treated as a hub id and fails
    # fast (invalid model id), landing on the heuristic fallback.
    counter = get_token_counter(str(tmp_path / "nope" / "tokenizer.json"))
    assert isinstance(counter, HeuristicTokenCounter)


def test_heuristic_counts_scale_up():
    # Two words plus punctuation, scaled by 1.3 and floored, plus one.
    assert HeuristicTokenCounter().count("hello, world") >= 3
