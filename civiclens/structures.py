from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

from civiclens.nlp import terms

MIN_DUPLICATE_SIMILARITY = 0.48
DEFAULT_AREAS = (
    "Central District",
    "Greenfield",
    "Harbor Area",
    "Northside",
    "Old Town",
    "Riverside",
    "West End",
)


class ReportLike(Protocol):
    id: str
    description: str
    area: str
    category: str
    created_at: datetime


@dataclass(frozen=True)
class DuplicateMatch:
    issue_id: str
    description: str
    similarity: float


class InvertedIndex:
    """Indexes terms to report IDs, then scores only matching candidates."""

    def __init__(self, reports: Iterable[ReportLike]) -> None:
        self.reports: dict[str, ReportLike] = {}
        self.term_counts: dict[str, Counter[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)

        for report in reports:
            counts = Counter(terms(report.description))
            self.reports[report.id] = report
            self.term_counts[report.id] = counts
            for term in counts:
                self.postings[term].add(report.id)

    def find_similar(
        self,
        description: str,
        area: str,
        category: str,
        limit: int = 3,
    ) -> list[DuplicateMatch]:
        query_counts = Counter(terms(description))
        candidate_ids: set[str] = set()
        for term in query_counts:
            candidate_ids.update(self.postings.get(term, ()))

        query_area = area.casefold()
        query_weights: dict[str, float] = {}
        for term, count in query_counts.items():
            document_frequency = len(self.postings.get(term, ()))
            inverse_frequency = math.log(
                (len(self.reports) + 1) / (document_frequency + 1)
            ) + 1
            query_weights[term] = count * inverse_frequency
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return []

        matches: list[DuplicateMatch] = []
        for issue_id in candidate_ids:
            report = self.reports[issue_id]
            if report.area.casefold() != query_area or report.category != category:
                continue

            report_weights: dict[str, float] = {}
            for term, count in self.term_counts[issue_id].items():
                document_frequency = len(self.postings[term])
                inverse_frequency = math.log(
                    (len(self.reports) + 1) / (document_frequency + 1)
                ) + 1
                report_weights[term] = count * inverse_frequency

            dot_product = sum(
                query_weights[term] * report_weights[term]
                for term in query_weights.keys() & report_weights.keys()
            )
            report_norm = math.sqrt(
                sum(weight * weight for weight in report_weights.values())
            )
            similarity = dot_product / (query_norm * report_norm)
            if similarity >= MIN_DUPLICATE_SIMILARITY:
                matches.append(
                    DuplicateMatch(report.id, report.description, similarity)
                )

        return sorted(matches, key=lambda match: match.similarity, reverse=True)[
            :limit
        ]


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.value: str | None = None


class AreaTrie:
    def __init__(self, areas: Iterable[str] = ()) -> None:
        self.root = _TrieNode()
        for area in areas:
            self.add(area)

    def add(self, area: str) -> None:
        normalized = area.strip()
        if not normalized:
            return
        node = self.root
        for character in normalized.casefold():
            node = node.children.setdefault(character, _TrieNode())
        node.value = normalized

    def suggest(self, prefix: str, limit: int = 8) -> list[str]:
        node = self.root
        for character in prefix.strip().casefold():
            node = node.children.get(character)
            if node is None:
                return []

        suggestions: list[str] = []
        pending = [node]
        while pending and len(suggestions) < limit:
            current = pending.pop()
            if current.value is not None:
                suggestions.append(current.value)
            pending.extend(
                child
                for _, child in sorted(current.children.items(), reverse=True)
            )
        return sorted(suggestions, key=str.casefold)[:limit]


PRIORITY_BY_CATEGORY = {
    "public_safety": 100,
    "drainage": 70,
    "water": 65,
    "roads": 55,
    "lighting": 45,
    "waste": 35,
}
URGENT_TERMS = frozenset(
    {"accident", "collapse", "danger", "fire", "injured", "leak", "live", "smoke"}
)


def priority_score(report: ReportLike, now: datetime | None = None) -> int:
    current_time = now or datetime.now(timezone.utc)
    created_at = report.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max(0, int((current_time - created_at).total_seconds() // 86400))
    urgent_bonus = 30 if URGENT_TERMS.intersection(terms(report.description)) else 0
    return (
        PRIORITY_BY_CATEGORY.get(report.category, 25)
        + urgent_bonus
        + min(age_days, 30)
    )


def prioritize(
    reports: Iterable[ReportLike], now: datetime | None = None
) -> list[tuple[int, ReportLike]]:
    queue: list[tuple[int, float, str, ReportLike]] = []
    for report in reports:
        created_at = report.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        heapq.heappush(
            queue,
            (
                -priority_score(report, now),
                created_at.timestamp(),
                report.id,
                report,
            ),
        )

    prioritized_reports: list[tuple[int, ReportLike]] = []
    while queue:
        negative_score, _, _, report = heapq.heappop(queue)
        prioritized_reports.append((-negative_score, report))
    return prioritized_reports
