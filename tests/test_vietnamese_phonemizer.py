"""Tests for Vietnamese phonemization (sea-g2p -> IPA + tones)."""

import pytest

from piper.config import PhonemeType
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids

pytest.importorskip("sea_g2p")

from piper.phonemize_vietnamese import VietnamesePhonemizer  # noqa: E402

_TEXTS = [
    "Chương hai nghìn tám trăm mười. Bầy Gương.",
    "Thiếu niên ngước nhìn trời, uể oải nói: Tôi không biết!",
    "Nhà ông Hùng ở số 25, đường Lê Lợi, lúc 8h30 sáng.",
    "Ok, hello world?",
]


@pytest.fixture(name="phonemizer", scope="module")
def phonemizer_fixture() -> VietnamesePhonemizer:
    return VietnamesePhonemizer()


def test_sentences_are_split(phonemizer: VietnamesePhonemizer) -> None:
    result = phonemizer.phonemize(_TEXTS[0])
    assert len(result) == 2


def test_numbers_are_spelled_out(phonemizer: VietnamesePhonemizer) -> None:
    result = phonemizer.phonemize("25")
    assert result and not any(c.isdigit() and c not in "123456" for c in result[0])
    assert len(result[0]) > 3


def test_empty_input(phonemizer: VietnamesePhonemizer) -> None:
    assert phonemizer.phonemize("") == []
    assert phonemizer.phonemize("   \n ") == []


def test_no_leading_or_trailing_space(phonemizer: VietnamesePhonemizer) -> None:
    for text in _TEXTS:
        for sentence in phonemizer.phonemize(text):
            assert sentence[0] != " " and sentence[-1] != " "


def test_every_phoneme_has_an_id(phonemizer: VietnamesePhonemizer) -> None:
    for text in _TEXTS:
        for sentence in phonemizer.phonemize(text):
            for phoneme in sentence:
                assert len(phoneme) == 1
                assert phoneme in DEFAULT_PHONEME_ID_MAP


def test_phonemes_to_ids(phonemizer: VietnamesePhonemizer) -> None:
    sentence = phonemizer.phonemize(_TEXTS[0])[0]
    ids = phonemes_to_ids(sentence, DEFAULT_PHONEME_ID_MAP)
    assert len(ids) > len(sentence)


def test_phoneme_type_enum() -> None:
    assert PhonemeType("vietnamese") == PhonemeType.VIETNAMESE
