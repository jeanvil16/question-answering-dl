"""Rule-based tokenization for extractive QA.

Why a custom lightweight tokenizer?
-----------------------------------
The model needs to map an answer span (character offsets in the raw context)
to *token* positions so it can be trained to predict start/end indices, and to
map predicted token positions back to characters so the final answer can be
extracted from the original text. Standard subword tokenizers (WordPiece,
BPE) make this mapping harder for a small academic project, while this regex
tokenizer keeps a simple 1:1 token -> character-offset alignment.

The vocabulary is built on lowercased text, which reduces the number of
distinct tokens a small dataset has to learn.
"""

import re
from dataclasses import dataclass

# A token is either:
#   1. a number (supports "1,000", "8848", "72.5", "71%"),
#   2. a word starting with a letter (may contain digits, apostrophes,
#      hyphens: "don't", "state-of-the-art", "TCP/IP" splits into TCP / / IP),
#   3. any single non-space symbol (punctuation), kept as its own token.
TOKEN_RE = re.compile(
    r"""
    \d+(?:,\d{3})*(?:\.\d+)?%?     # numbers: 1,000 / 8,848 / 72.5 / 71%
  | [A-Za-z][A-Za-z0-9_'\-]*       # words incl. don't / well-known
  | [^\sA-Za-z0-9_]                # single punctuation / symbol characters
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    """One token plus its character span inside the ORIGINAL string."""

    text: str   # lowercased surface form used by the model
    start: int  # inclusive char offset in the original string
    end: int    # exclusive char offset in the original string


def tokenize(text):
    """Return the list of lowercase token strings for `text`."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def tokenize_with_offsets(text):
    """Tokenize while remembering each token's position in the source string.

    The character offsets are what allows us to:
      * convert SQuAD's `answer_start` (a char index) into token indices, and
      * slice the predicted token span back out of the raw context at
        inference time.
    """
    return [
        Token(match.group(0).lower(), match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
    ]
