from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from civiclens.training_data import TRAINING_EXAMPLES

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "be", "beside", "by", "for", "from",
        "has", "have", "in", "is", "it", "near", "of", "on", "or", "the",
        "to", "was", "were", "with",
    }
)


def terms(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in STOP_WORDS
    ]


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float


class NaiveBayesIssueClassifier:
    """Small, explainable multinomial Naive Bayes model trained from examples."""

    def __init__(
        self, examples: Iterable[tuple[str, str]] = TRAINING_EXAMPLES
    ) -> None:
        class_documents: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()

        for text, category in examples:
            document_terms = terms(text)
            if not document_terms:
                continue
            class_documents[category] += 1
            token_counts[category].update(document_terms)
            vocabulary.update(document_terms)

        if not class_documents or not vocabulary:
            raise ValueError("The classifier requires at least one labeled example.")

        self.categories = tuple(sorted(class_documents))
        self.vocabulary_size = len(vocabulary)
        self.class_documents = class_documents
        self.token_counts = token_counts
        self.vocabulary = vocabulary
        self.total_documents = sum(class_documents.values())

    def predict(self, text: str) -> Prediction:
        document_terms = terms(text)
        scores: dict[str, float] = {}
        for category in self.categories:
            score = math.log(
                self.class_documents[category] / self.total_documents
            )
            total_tokens = sum(self.token_counts[category].values())
            denominator = total_tokens + self.vocabulary_size
            for term, count in Counter(document_terms).items():
                if term in self.vocabulary:
                    probability = (
                        self.token_counts[category][term] + 1
                    ) / denominator
                    score += count * math.log(probability)
            scores[category] = score

        highest = max(scores.values())
        exponentials = {
            category: math.exp(score - highest)
            for category, score in scores.items()
        }
        total = sum(exponentials.values())
        category = max(scores, key=scores.__getitem__)
        return Prediction(category, exponentials[category] / total)
