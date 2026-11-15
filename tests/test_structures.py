from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from civiclens.structures import AreaTrie, InvertedIndex, prioritize


@dataclass
class Report:
    id: str
    description: str
    area: str
    category: str
    created_at: datetime


def test_inverted_index_finds_similar_reports_in_same_area_and_category() -> None:
    old_report = Report(
        "one",
        "Large pothole on Oak Street near the school needs repair",
        "Northside",
        "roads",
        datetime.now(timezone.utc),
    )
    index = InvertedIndex(
        [
            old_report,
            Report(
                "other-area",
                old_report.description,
                "West End",
                "roads",
                old_report.created_at,
            ),
            Report(
                "other-category",
                old_report.description,
                "Northside",
                "waste",
                old_report.created_at,
            ),
        ]
    )

    matches = index.find_similar(
        "Pothole on Oak Street near the school needs fixing",
        "Northside",
        "roads",
    )

    assert [match.issue_id for match in matches] == ["one"]
    assert matches[0].similarity > 0.48


def test_area_trie_returns_prefix_matches_without_case_sensitivity() -> None:
    trie = AreaTrie(["Northside", "North Park", "West End"])

    assert trie.suggest("north") == ["North Park", "Northside"]
    assert trie.suggest("unknown") == []


def test_priority_queue_puts_safety_reports_before_routine_reports() -> None:
    now = datetime.now(timezone.utc)
    routine = Report("routine", "Trash pickup is late", "Northside", "waste", now)
    safety = Report(
        "safety",
        "A live electrical wire is hanging over the path",
        "Northside",
        "public_safety",
        now - timedelta(days=2),
    )

    ranked = prioritize([routine, safety], now)

    assert ranked[0][1].id == "safety"
    assert ranked[0][0] > ranked[1][0]
