# CivicLens AI

**Community reports in. A review-ready action queue out.**

CivicLens turns a short, plain-language civic report into a suggested service category, nearby duplicate matches, and a priority-ranked triage queue. It is designed to help a human operations team handle noisy reports faster; it does not automatically close, merge, or dispatch requests.

## Why it is useful

Residents often report the same blocked drain, broken light, or road hazard in different words. Teams then spend time sorting reports before they can act. CivicLens provides a small, explainable first pass:

1. A resident submits a description and neighborhood, without needing an account.
2. A lightweight Naive Bayes NLP model suggests a category and confidence.
3. An inverted index finds likely duplicates in the same area and category.
4. A heap-based queue ranks safety-sensitive and older reports for review.
5. Low-confidence reports and possible duplicates stay visible for a person to decide.

The demo avoids collecting names, email addresses, or phone numbers. Do not put personal or emergency information in report descriptions.

## Tech and engineering

- **Python / FastAPI:** typed, validated REST API with interactive OpenAPI docs.
- **AI / NLP:** an interpretable multinomial Naive Bayes text classifier, trained on a small, included example dataset. The confidence is a routing hint, not a calibrated probability.
- **DSA:** TF-IDF-style inverted index for candidate duplicate search, a trie for area prefix suggestions, and `heapq` for priority ordering.
- **Storage:** SQLAlchemy with SQLite locally and PostgreSQL in the cloud.
- **Cloud:** Docker image and Render Blueprint for a web service plus managed PostgreSQL.
- **Quality:** focused tests and GitHub Actions CI.

## Run locally

Requires Python 3.11 or newer.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m uvicorn civiclens.api:app --reload
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m uvicorn civiclens.api:app --reload
```

The service is at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`. By default, data is stored in `civiclens.db`. Set `CIVICLENS_DATABASE_URL` to use another SQLAlchemy database URL.

## Example output

Submit a road issue and then a differently worded report about the same issue in the same area. CivicLens suggests the `roads` category for both and marks the second as `possible_duplicate`, with a similarity score. The sample run is in [`sample-output.json`](sample-output.json).

The project currently provides a REST API and its interactive docs, not a separate visual dashboard. The `/docs` page lets you try each endpoint in your browser.

## Try it

```powershell
curl.exe -X POST http://127.0.0.1:8000/issues `
  -H "Content-Type: application/json" `
  -d '{"description":"Large pothole on Oak Street near the school needs repair","area":"Northside"}'
```

Submit a similar report in the same area to see a duplicate candidate. Other useful endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/healthz` | Container health check |
| `POST` | `/issues` | Classify and save a report |
| `GET` | `/issues` | Browse reports; filter by exact `area` or `category` |
| `GET` | `/issues/{id}` | View one report |
| `GET` | `/triage/queue` | Get reports in priority order |
| `GET` | `/areas/suggest?q=north` | Prefix suggestions from the area trie |

Run the test suite:

```powershell
python -m pytest
```

## Deploy to Render

1. Push this directory as its own GitHub repository.
2. In Render, choose **New → Blueprint** and connect that repository.
3. Render reads `render.yaml` to create the Docker web service and PostgreSQL database.
4. After deployment, check `/healthz` and open `/docs`.

The database and web service may incur charges according to the provider's current plan. Review the plan before creating cloud resources. For a container-only smoke test:

```powershell
docker build -t civiclens-ai .
docker run --rm -p 8000:8000 civiclens-ai
```

## DSA and model notes

- Duplicate lookup builds term postings once per submitted report from up to the latest 5,000 stored issues. Candidate scoring uses cosine similarity with inverse-document-frequency weights; it checks only reports sharing terms and then filters by area and predicted category.
- The priority queue gives safety reports a higher starting score and adds a bounded age bonus. It is a transparent triage heuristic, not a substitute for emergency dispatch rules.
- The included training examples make the project runnable without downloading a model. They are demonstration data, not a representative civic dataset. Review model suggestions; expand and evaluate the dataset before real use.
- Area suggestions use a trie seeded with demo neighborhood names and add observed report areas.

## Production hardening

This is a portfolio MVP, not a municipal production service. Before real deployment, add authentication and authorization, abuse/rate controls, privacy review and retention rules, audit logging, model evaluation with representative data, database migrations, monitoring, and an operational emergency-reporting policy. Multiple replicas also need a shared search/index strategy rather than rebuilding the duplicate index per request.
