# app/bigram_model.py
from __future__ import annotations
import re
import random
from collections import defaultdict, Counter
from typing import Iterable, List, Dict

# Light tokenizer: only keep letters and apostrophes, lowercase
_TOKEN_RE = re.compile(r"[A-Za-z']+")

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BigramModel:
    """
    A minimal bigram model:
    - Use Counter to count the frequency of each previous word -> next word
    - Generate by weighted random sampling
    - If no available transition, fall back to the most common word, avoid getting stuck
    """
    def __init__(self, corpus: Iterable[str] | None = None, seed: int | None = 42) -> None:
        if seed is not None:
            random.seed(seed)
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.start_words: list[str] = []
        self._fallbacks: list[str] = []

        if corpus:
            self.train(corpus)

    # Can be called repeatedly to append/retrain
    def train(self, corpus: Iterable[str]) -> None:
        self.bigrams.clear()
        self.vocab.clear()
        self.start_words.clear()

        for line in corpus:
            toks = _tokenize(line)
            if not toks:
                continue
            self.vocab.update(toks)
            self.start_words.append(toks[0])
            for w1, w2 in zip(toks, toks[1:]):
                self.bigrams[w1][w2] += 1

        # As fallback candidates: Top-N most common word pairs
        global_top = Counter()
        for c in self.bigrams.values():
            global_top.update(c)
        self._fallbacks = [w for w, _ in global_top.most_common(10)]

    def next_word(self, cur: str) -> str | None:
        # Given the current word, sample the next word by frequency; if no transition, fall back.
        cur = cur.lower()
        if cur in self.bigrams and self.bigrams[cur]:
            choices, weights = zip(*self.bigrams[cur].items())
            return random.choices(list(choices), list(weights), k=1)[0]
        return random.choice(self._fallbacks) if self._fallbacks else None

    def generate_text(self, start_word: str, length: int = 10) -> str:
        # Generate length words (including the start word) starting from start_word.
        if not self.vocab:
            return ""

        cur = (start_word or "").lower()
        if cur not in self.vocab:
            # If the start word is not in the vocabulary, use the first word of the training corpus as fallback.
            cur = random.choice(self.start_words) if self.start_words else random.choice(list(self.vocab))

        out = [cur]
        for _ in range(max(0, length - 1)):
            nxt = self.next_word(cur)
            if not nxt:
                break
            out.append(nxt)
            cur = nxt
        return " ".join(out)

    # Convenient constructor: create a model from a long text
    @classmethod
    def from_text(cls, text: str, **kw) -> "BigramModel":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return cls(lines, **kw)
