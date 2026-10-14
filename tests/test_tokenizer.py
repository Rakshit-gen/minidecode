import pytest

from minidecode.corpus import generate
from minidecode.tokenizer import CharTokenizer


def test_round_trip_on_logs():
    text = generate(200)
    tok = CharTokenizer.fit(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.size < 60
    assert tok.chars[0] == "\n"  # sorted, so newline comes first


def test_unknown_characters_and_duplicates():
    tok = CharTokenizer("ab")
    with pytest.raises(ValueError, match="'z' is not in the vocabulary"):
        tok.encode("az")
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer("aa")
