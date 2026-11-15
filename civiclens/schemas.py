from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReportCreate(BaseModel):
    description: str = Field(min_length=12, max_length=2000)
    area: str = Field(min_length=2, max_length=80)

    @field_validator("description", "area")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Value must not be blank.")
        return normalized


class DuplicateResponse(BaseModel):
    issue_id: str
    description: str
    similarity: float


class IssueResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    description: str
    area: str
    category: str
    confidence: float
    status: str
    duplicate_of: str | None
    created_at: datetime
    priority: int
    duplicates: list[DuplicateResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
