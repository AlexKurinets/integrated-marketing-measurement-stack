"""Shared fixtures. One small synthetic panel and one MMM fit are built per session and reused."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from measurement.pipeline.weekly import load_panel, run_refresh
from measurement.repository import InMemoryRepository, PostgresRepository
from seed.generate import generate

SMALL = {"n_markets": 20, "n_weeks": 60, "seed": 7, "tv_weeks": 3}


@pytest.fixture(scope="session")
def small_seed():
    return generate(**SMALL)


@pytest.fixture(scope="session")
def repo(small_seed):
    r = InMemoryRepository()
    r.load_inputs(small_seed)
    return r


@pytest.fixture(scope="session")
def panel(repo):
    return load_panel(repo)


@pytest.fixture(scope="session")
def refresh_summary(repo):
    """Runs the whole weekly refresh once (test profile) and publishes a run into the in-memory repo."""
    return run_refresh(repo, profile="test", seed=1, tv_weeks=3)


@pytest.fixture(scope="session")
def fitted_repo(repo, refresh_summary):
    return repo


@pytest.fixture(scope="session")
def db_repo():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; database tests run in CI against the Postgres service")
    engine = create_engine(url, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not reachable: {exc}")
    r = PostgresRepository(engine=engine)
    r.migrate()
    r.reset()
    return r
