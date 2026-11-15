import asyncio

import httpx
from fastapi import FastAPI

from civiclens.api import create_app


def request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    json: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, json=json, params=params)

    return asyncio.run(send())


def test_report_flow_flags_duplicates_and_exposes_triage_queue() -> None:
    app = create_app("sqlite://")

    first = request(
        app,
        "POST",
        "/issues",
        json={
            "description": "Large pothole on Oak Street near the school needs repair",
            "area": "Northside",
        },
    )
    assert first.status_code == 201
    first_report = first.json()
    assert first_report["category"] == "roads"
    assert first_report["status"] == "open"

    second = request(
        app,
        "POST",
        "/issues",
        json={
            "description": "Pothole on Oak Street near the school needs fixing",
            "area": "Northside",
        },
    )
    assert second.status_code == 201
    duplicate = second.json()
    assert duplicate["status"] == "possible_duplicate"
    assert duplicate["duplicate_of"] == first_report["id"]
    assert duplicate["duplicates"][0]["similarity"] > 0.48

    queue = request(app, "GET", "/triage/queue")
    assert queue.status_code == 200
    assert len(queue.json()) == 2


def test_unclassified_report_is_kept_for_human_review() -> None:
    app = create_app("sqlite://")

    response = request(
        app,
        "POST",
        "/issues",
        json={
            "description": "The purple moon sings quietly",
            "area": "Northside",
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "needs_review"


def test_health_validation_and_area_suggestions() -> None:
    app = create_app("sqlite://")

    assert request(app, "GET", "/healthz").json() == {"status": "ok"}
    assert request(
        app, "POST", "/issues", json={"description": "short", "area": "X"}
    ).status_code == 422
    assert request(app, "GET", "/areas/suggest", params={"q": "north"}).json() == [
        "Northside"
    ]
