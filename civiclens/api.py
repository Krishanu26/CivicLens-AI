from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from civiclens.nlp import NaiveBayesIssueClassifier
from civiclens.schemas import (
    DuplicateResponse,
    HealthResponse,
    IssueResponse,
    ReportCreate,
)
from civiclens.storage import IssueRecord, create_database
from civiclens.structures import (
    DEFAULT_AREAS,
    AreaTrie,
    InvertedIndex,
    prioritize,
    priority_score,
)

LOW_CONFIDENCE_THRESHOLD = 0.38
MAX_DUPLICATE_SCAN = 5000


def create_app(database_url: str | None = None) -> FastAPI:
    engine, session_factory = create_database(database_url)
    classifier = NaiveBayesIssueClassifier()
    area_trie = AreaTrie(DEFAULT_AREAS)
    app = FastAPI(
        title="CivicLens AI",
        description=(
            "Explainable triage for community reports: classify, find similar "
            "reports, and rank work for human review."
        ),
        version="0.1.0",
    )

    def get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    def serialize_issue(
        record: IssueRecord,
        duplicates: list[DuplicateResponse] | None = None,
    ) -> IssueResponse:
        created_at = record.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return IssueResponse(
            id=record.id,
            description=record.description,
            area=record.area,
            category=record.category,
            confidence=record.confidence,
            status=record.status,
            duplicate_of=record.duplicate_of,
            created_at=created_at,
            priority=record.priority,
            duplicates=duplicates or [],
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post(
        "/issues",
        response_model=IssueResponse,
        status_code=201,
        tags=["reports"],
    )
    def create_issue(
        payload: ReportCreate, session: Session = Depends(get_session)
    ) -> IssueResponse:
        prediction = classifier.predict(payload.description)
        existing = session.scalars(
            select(IssueRecord)
            .order_by(IssueRecord.created_at.desc())
            .limit(MAX_DUPLICATE_SCAN)
        ).all()
        matches = InvertedIndex(existing).find_similar(
            payload.description, payload.area, prediction.category
        )

        if matches:
            status = "possible_duplicate"
        elif prediction.confidence < LOW_CONFIDENCE_THRESHOLD:
            status = "needs_review"
        else:
            status = "open"

        record = IssueRecord(
            description=payload.description,
            area=payload.area,
            category=prediction.category,
            confidence=prediction.confidence,
            status=status,
            duplicate_of=matches[0].issue_id if matches else None,
            created_at=datetime.now(timezone.utc),
            priority=0,
        )
        record.priority = priority_score(record)
        session.add(record)
        session.commit()
        session.refresh(record)
        duplicate_responses = [
            DuplicateResponse(
                issue_id=match.issue_id,
                description=match.description,
                similarity=match.similarity,
            )
            for match in matches
        ]
        return serialize_issue(record, duplicate_responses)

    @app.get("/issues", response_model=list[IssueResponse], tags=["reports"])
    def list_issues(
        area: str | None = None,
        category: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
        session: Session = Depends(get_session),
    ) -> list[IssueResponse]:
        statement = select(IssueRecord).order_by(IssueRecord.created_at.desc())
        if area:
            statement = statement.where(IssueRecord.area.ilike(area.strip()))
        if category:
            statement = statement.where(IssueRecord.category == category.strip())
        records = session.scalars(statement.limit(limit)).all()
        return [serialize_issue(record) for record in records]

    @app.get("/triage/queue", response_model=list[IssueResponse], tags=["triage"])
    def triage_queue(
        limit: int = Query(default=20, ge=1, le=100),
        session: Session = Depends(get_session),
    ) -> list[IssueResponse]:
        records = session.scalars(select(IssueRecord)).all()
        prioritized = prioritize(records)[:limit]
        result: list[IssueResponse] = []
        for score, record in prioritized:
            response = serialize_issue(record)
            response.priority = score
            result.append(response)
        return result

    @app.get("/issues/{issue_id}", response_model=IssueResponse, tags=["reports"])
    def get_issue(
        issue_id: str, session: Session = Depends(get_session)
    ) -> IssueResponse:
        record = session.get(IssueRecord, issue_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Issue not found.")
        return serialize_issue(record)

    @app.get("/areas/suggest", response_model=list[str], tags=["areas"])
    def suggest_areas(
        q: str = Query(min_length=1, max_length=80),
        limit: int = Query(default=8, ge=1, le=20),
        session: Session = Depends(get_session),
    ) -> list[str]:
        trie = AreaTrie(DEFAULT_AREAS)
        for area in session.scalars(select(IssueRecord.area).distinct()):
            trie.add(area)
        return trie.suggest(q, limit)

    return app


app = create_app()
