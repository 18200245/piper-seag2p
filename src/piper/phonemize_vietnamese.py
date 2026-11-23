"""Vietnamese phonemization: sea-g2p text normalization + G2P -> IPA + tones.

sea-g2p (https://github.com/pnnbao97/sea-g2p) has a Rust core. Its pipeline
spells out numbers, dates, currency and abbreviations, then returns IPA with
stress marks and tone symbols. Every symbol it emits for Vietnamese already has
an id in :data:`piper.phoneme_ids.DEFAULT_PHONEME_ID_MAP`, so Vietnamese voices
stay compatible with the IPA-based (espeak) warmstart.

Requires ``pip install sea-g2p`` (the ``vi`` extra).
"""

import logging
import re
import unicodedata
from typing import List

from .phoneme_ids import DEFAULT_PHONEME_ID_MAP

_LOGGER = logging.getLogger(__name__)

# Split on explicit terminal punctuation followed by space, and on newlines.
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


class VietnamesePhonemizer:
    """Convert Vietnamese text to IPA phonemes with sea-g2p."""

    def __init__(self, lang: str = "vi") -> None:
        from sea_g2p import SEAPipeline

        self._pipeline = SEAPipeline(lang=lang)

    def phonemize(self, text: str) -> List[List[str]]:
        """Return phonemes (one codepoint each), grouped by sentence."""
        text = unicodedata.normalize("NFC", text)
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return []

        ipa_sentences = self._pipeline.run(sentences)
        if isinstance(ipa_sentences, str):
            ipa_sentences = [ipa_sentences]

        all_phonemes: List[List[str]] = []
        for sentence, ipa in zip(sentences, ipa_sentences):
            phonemes: List[str] = []
            for char in unicodedata.normalize("NFC", ipa):
                if char in DEFAULT_PHONEME_ID_MAP:
                    phonemes.append(char)
                elif char.isspace():
                    if not phonemes or phonemes[-1] != " ":
                        phonemes.append(" ")
                else:
                    _LOGGER.warning(
                        "Dropping unknown symbol %r (U+%04X) from: %s",
                        char,
                        ord(char),
                        sentence,
                    )

            while phonemes and phonemes[0] == " ":
                phonemes.pop(0)
            while phonemes and phonemes[-1] == " ":
                phonemes.pop()

            if phonemes:
                all_phonemes.append(phonemes)

        return all_phonemes
