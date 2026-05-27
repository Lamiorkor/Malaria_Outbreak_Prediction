# CI/CD Implementation — Walkthrough

This document explains every file added to the malaria-outbreak project to
implement Continuous Integration and Continuous Deployment as described in the
SW10 lecture (`ARTIFIN_SW10.pdf`).  It is intended to be read top-to-bottom by
whoever inherits the project.

---

## 1.  What we are building and why

The lecture defines the pipeline (Slide 6 — "Pipeline Overview") as:

```
Push → Run → Build → Push
 │      │      │      │
 │      │      │      └─ optional: push image to registry (GHCR)
 │      │      └──────── CD: build Docker image
 │      └─────────────── CI: run pytest in clean environment
 └────────────────────── push code to GitHub
```

So far the project has:

| Layer                | Status before this work                              |
| -------------------- | ---------------------------------------------------- |
| Training pipeline    | ✅ `training_pipeline/train_pipeline.py` + MLflow log |
| Inference API        | ✅ FastAPI in `dockerisation_and_deployment/webservices` |
| Dockerfile           | ✅ Present                                            |
| Unit tests           | ⚠️ Exist, but co-located with code; one of them needs a real model |
| **CI/CD automation** | ❌ Missing — everything was manual                    |

The job of this work is to plug that last gap **without rewriting any of the
existing code**.  We treat the existing code as a contract and wrap automation
around it.

---

## 2.  Final file structure

Files marked **NEW** are added by this work.  Everything else already existed.

```
FINAL_FOLDER/
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 ← NEW   GitHub Actions pipeline
│
├── .dockerignore                     ← NEW   Smaller, faster Docker builds
├── docker-compose.yml                ← NEW   Local end-to-end stack (API+MLflow+DB)
├── pytest.ini                        ← NEW   Test discovery configuration
├── requirements.txt                  (unchanged)
├── requirements-dev.txt              ← NEW   Tooling kept out of the runtime image
│
├── tests/                            ← NEW   Centralised, CI-safe test folder
│   ├── __init__.py
│   ├── conftest.py                   ← NEW   Makes source folders importable
│   ├── test_prediction_core.py       ← NEW   Slide 9–11 dummy-model unit tests
│   ├── test_api_contract.py          ← NEW   Pydantic schema validation
│   └── test_smoke_pipeline.py        ← NEW   End-to-end pipeline with injected dummy
│
├── data/                             (unchanged — excluded from image)
├── models/                           (unchanged — baked into image)
├── mlruns/                           (unchanged — mounted by compose, never in image)
├── monitoring/                       (unchanged)
├── training_pipeline/                (unchanged)
│   └── train_pipeline.py
└── dockerisation_and_deployment/
    ├── webservices/
    │   ├── main.py                   (unchanged)
    │   ├── predict.py                (unchanged)
    │   ├── prediction_core.py        (unchanged)
    │   ├── Dockerfile                (unchanged)
    │   ├── test_predict.py           ⚠️  see §6 — recommend renaming
    │   └── test_prediction_core.py   ⚠️  see §6 — superseded by tests/
    └── batch/                        (unchanged)
```

---

## 3.  File-by-file explanation

### 3.1 `.github/workflows/ci-cd.yml` — the actual pipeline

This is the YAML that GitHub Actions reads.  It defines two **jobs**:

#### Job 1: `test` (the CI)

Maps to Slide 7 — "CI in Practice":

| Slide bullet                       | What the YAML does                          |
| ---------------------------------- | ------------------------------------------- |
| Runs on clean Ubuntu machine       | `runs-on: ubuntu-latest`                    |
| Installs dependencies              | `pip install -r requirements.txt -r requirements-dev.txt` (we use pip rather than Poetry to match the existing `requirements.txt`; see §7) |
| Executes pytest tests              | `pytest -v --tb=short --maxfail=1`          |
| Fails pipeline if tests fail       | `--maxfail=1` and non-zero exit propagate   |

**Triggers** — when does it run?

| Event                               | What runs                                |
| ----------------------------------- | ---------------------------------------- |
| Push to `main` or `develop`         | tests + build + push image               |
| Pull request targeting `main`       | tests only (no push — quality gate)      |
| Manual ("Run workflow" button)      | full pipeline                            |

The `concurrency:` block cancels the previous run on the same branch when a
new commit lands — so a busy PR doesn't queue up four redundant builds.

#### Job 2: `build-and-push` (the CD)

This depends on (`needs: test`) the CI job — so it **only runs if tests pass**.
That gate is the whole point of the test/build separation.

| Step                       | Purpose                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `docker/setup-buildx-action` | Modern builder with GitHub-Actions layer caching            |
| `docker/login-action`      | Logs in to GHCR using the automatic `GITHUB_TOKEN` (no PAT needed) |
| `docker/metadata-action`   | Computes tags like `sha-a1b2c3d`, `main`, `latest`            |
| `docker/build-push-action` | Builds image; pushes ONLY on `push` to `main`                 |
| Smoke test                 | `docker run` the image, `curl /`, fail the job if unhealthy   |

The smoke test is critical because it catches a class of bug that unit tests
cannot: things that break only inside the container (wrong working directory,
missing system library, broken `CMD`).

**Why GHCR specifically?**  Because every GitHub repo already has a
free GHCR namespace at `ghcr.io/<owner>/<repo>` with auth wired in via
`GITHUB_TOKEN`.  No additional secrets, no Docker Hub rate limits.

### 3.2 `.dockerignore` — keep the image lean

Slide 12 ("CD: Docker Build") says Docker should "package code + dependencies"
and "create a reproducible runtime."  Without `.dockerignore`, the entire repo
context is sent to the Docker daemon — including the 4-MB `mlruns/` folder,
the 612-KB `mlflow.db`, every Jupyter notebook, `__pycache__`, and the raw
training data.  Result:

* **slower builds** (transfer + hash of huge context),
* **bigger images** (more layers),
* **worse caching** (any notebook edit invalidates the build cache).

`.dockerignore` excludes everything that is not strictly needed at runtime.
For this project, the runtime only needs:

* `dockerisation_and_deployment/webservices/` (the API code)
* `models/` (pickle + scaler + metadata)
* `requirements.txt`

Everything else stays on the developer's machine and in Git history.

### 3.3 `pytest.ini` — test discovery rules

This is a small file with a big effect.  Without it, `pytest` from the repo
root will discover **every** `test_*.py` it can find, including:

* `dockerisation_and_deployment/webservices/test_predict.py`, which calls
  `PredictionPipeline()` at import time and crashes in CI because the model
  files might not be available everywhere.
* `dockerisation_and_deployment/webservices/test_prediction_core.py`, which
  is fine but is a duplicate of what now lives under `tests/`.

`pytest.ini` pins discovery to **`testpaths = tests`**, so only the curated
CI test suite runs.  This is the cleanest way to honour Slide 11's "Avoids
MLflow dependency in CI" — those imports never even happen.

### 3.4 `requirements-dev.txt` — dev/test tooling

`pytest`, `httpx` (FastAPI test client) and `pytest-cov` are needed by CI but
**not** by the production container.  Keeping them in a separate file means:

* the runtime image stays small,
* CI explicitly states what test tooling it relies on,
* the same dev file works on developer laptops.

### 3.5 `tests/conftest.py` — making imports work

`pytest` runs from the repo root, but our test files want to do
`from prediction_core import predict`.  Without help, Python will not find
that module because it lives in `dockerisation_and_deployment/webservices/`.

`conftest.py` is auto-loaded by `pytest` before any test.  It prepends the
right directories to `sys.path` so tests can do `import predict`,
`import main`, etc., without us having to install the project as a package.
This is a deliberate trade-off: a real-world project would ship a
`pyproject.toml` with a proper package and `pip install -e .` it, but for a
course project this is simpler and changes nothing about how the existing code
runs.

### 3.6 `tests/test_prediction_core.py` — Slide 9–11 in code

Each test maps directly to a bullet from the slides:

| Slide | Bullet                              | Test function                              |
| ----- | ----------------------------------- | ------------------------------------------ |
| 9     | Finds functions starting with `test_` | every function name begins with `test_`  |
| 9     | Executes tests and checks assertions | `assert` statements throughout            |
| 9     | Outputs PASS or FAIL                | pytest's own runner                       |
| 10    | Input validation (shape, types)     | `test_rejects_wrong_feature_count`, `test_rejects_extra_features`, `test_rejects_non_numeric_value`, `test_rejects_non_sequence_input` |
| 10    | Prediction logic                    | `test_valid_prediction_returns_model_output`, `test_model_receives_correct_shape` |
| 10    | Output format                       | implicit in the assertions on return value |
| 10    | Error handling                      | `test_invalid_input_never_calls_model`    |
| 11    | Replaces real ML model              | `DummyModel`, `ShapeCheckingDummyModel`   |
| 11    | Avoids MLflow dependency in CI      | nothing in the test file imports `mlflow` |
| 11    | Fast and stable testing             | the whole file runs in ~50 ms              |

### 3.7 `tests/test_api_contract.py` — protect the public API

The `CountryRecord` Pydantic schema in `main.py` is the contract every API
client depends on.  If someone accidentally removes the `ge=2000` bound on
`year` or changes `pop_density` to allow zero, every downstream consumer
breaks silently.  These tests catch that **the day the change is committed**,
because the CI will fail before the merge.

We deliberately do **not** start the FastAPI app here — we just import the
`CountryRecord` class.  This keeps the test under 100 ms and avoids
the lifespan event that would try to load the real model.

### 3.8 `tests/test_smoke_pipeline.py` — end-to-end with dummies

The unit tests prove individual functions work; this file proves they work
**together**.  It uses `monkeypatch` to swap out `_load_local` with a function
that injects a `DummyModel` and an `IdentityScaler`.  The real
`engineer_features` code then runs against realistic input — catching:

* missing columns in the feature engineering output,
* changes to the schema of `predict_single`'s return value,
* NaN leakage,
* mis-ordered feature columns.

If any of those bugs ever land, CI is red.

### 3.9 `docker-compose.yml` — the deployment story

The PDF's "Final Architecture" slide shows MLflow, the API, and (implicitly) a
prediction log database working together.  `docker-compose.yml` runs that
exact picture on a single machine:

```
                 ┌─────────────┐
                 │   MLflow    │  ← serves models on :5000
                 └──────▲──────┘
                        │  MLFLOW_TRACKING_URI
                        │
┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐
│   Client    │─▶│ malaria-api │─▶│  Postgres   │
└─────────────┘  └─────────────┘  └─────────────┘
                    :8000             :5433
```

Run it with one command:

```bash
docker compose up --build
```

This is what you would demonstrate in person to show CD actually worked: pull
the image from GHCR (or build locally), bring up MLflow, hit `/predict`,
inspect the model in MLflow's UI at <http://localhost:5000>.

---

## 4.  Putting it on GitHub — first-time setup

1. Initialise a git repo (if you haven't already) and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Add CI/CD pipeline (SW10)"
   git branch -M main
   git remote add origin git@github.com:<your-org>/<your-repo>.git
   git push -u origin main
   ```

2. On GitHub:
   * Go to **Settings → Actions → General** and make sure
     "Workflow permissions" is set to **Read and write permissions**.
     This lets the CD job push images to GHCR using `GITHUB_TOKEN`.
   * Go to **Settings → Packages**. The first time the workflow pushes,
     a package called `malaria-api` will appear under your account.

3. Push a commit.  Open the **Actions** tab — you will see the workflow
   running.  Successful run looks like this:

   ```
   ✓ CI – Lint & Unit Tests          (1m 12s)
   ✓ CD – Build & Push Docker Image  (2m 04s)
   ```

4. Pull and run the image anywhere:
   ```bash
   docker pull ghcr.io/<your-org>/<your-repo>/malaria-api:latest
   docker run -p 8000:8000 ghcr.io/<your-org>/<your-repo>/malaria-api:latest
   ```

---

## 5.  What an MLflow-aware deployment looks like

`predict.py` currently loads the model from local pickles.  The PDF describes
an architecture where MLflow is the source of truth.  Two ways to evolve:

### Option A — keep file-based loading (current)

* Train locally → `models/logistic_regression.pkl` is committed to Git.
* `Dockerfile` `COPY`s `models/` into the image.
* CI/CD pipeline rebuilds the image whenever those files change.

This is exactly what the project does today.  Simple, reproducible, and the
image is fully self-contained.

### Option B — pull from the MLflow registry at startup

A future iteration of `predict.py` could read `MLFLOW_TRACKING_URI` and load
the model from the registry's `Staging` stage:

```python
import mlflow.pyfunc

def _load_from_mlflow(self):
    uri = f"models:/{MODEL_NAME}/Staging"
    self.model = mlflow.pyfunc.load_model(uri)
```

The Docker image stays the same; only the env var changes between
environments.  The compose file (§3.9) wires `MLFLOW_TRACKING_URI` for this.

Not changed in this work because it would touch `predict.py` —
which the project says is already working.  Documenting it here is enough.

---

## 6.  Existing files we **recommend** changing

These are the only places where the project's current code conflicts with
the new CI setup.  Suggested changes:

### `dockerisation_and_deployment/webservices/test_predict.py`

This file calls `PredictionPipeline()` at import time, which loads the real
model from disk.  It also `print`s rather than `assert`s, so even when it
runs it doesn't validate anything.  Two options:

* **Rename** it to `manual_check_predict.py` (so pytest never discovers it)
  and document it as a manual smoke test.
* **Or** delete it — its behaviour is now covered by
  `tests/test_smoke_pipeline.py`.

### `dockerisation_and_deployment/webservices/test_prediction_core.py`

This is the original of what now lives under `tests/`.  Recommended:
delete the old copy to keep one source of truth.

You don't *have* to do either change — `pytest.ini` already excludes both
files from discovery — but cleaning up avoids confusion later.

---

## 7.  Design choices worth flagging

### Why pip and not Poetry?

Slide 7 mentions Poetry.  The project already ships a `requirements.txt`,
and the Dockerfile uses `pip install -r requirements.txt`.  Switching to
Poetry would require:

* a `pyproject.toml` with the same dependencies,
* a new `poetry.lock`,
* changing the Dockerfile and possibly `predict.py` imports.

That's a lot of churn for no functional gain.  We kept pip — the CI workflow
caches pip downloads via `actions/setup-python`'s built-in `cache: pip`, so
performance is essentially identical.

### Why not run training in CI?

Two reasons:

1. The feature store + scaler are already committed under `data/feature_store/`,
   so retraining would be deterministic but slow.
2. Training, by design, belongs to a scheduled batch flow
   (`dockerisation_and_deployment/batch/`), not the developer-feedback loop.
   CI should be fast — under two minutes is the budget that keeps people
   actually looking at the result.

### Why GitHub Container Registry instead of Docker Hub?

* `GITHUB_TOKEN` is auto-provided to workflows — no secrets to configure.
* No anonymous-pull rate limits.
* Permissions are tied to the repo, so revoking access is trivial.
* It is the lecture's recommended choice (Slide 13).

---

## 8.  Checklist — what to verify before declaring this done

- [ ] `pytest` passes locally
- [ ] `git push` triggers the workflow in the **Actions** tab
- [ ] The CI job is green
- [ ] The CD job pushes an image to `ghcr.io/<owner>/<repo>/malaria-api`
- [ ] `docker pull` works from another machine
- [ ] `docker compose up` brings the stack up locally
- [ ] `curl http://localhost:8000/` returns `{"status": "healthy", ...}`
- [ ] `curl -X POST http://localhost:8000/predict ...` returns a prediction

If all eight pass, the pipeline laid out in Slide 6 of the SW10 deck is fully
implemented.
