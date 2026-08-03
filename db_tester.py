"""
routers/db_tester.py

FastAPI router that exposes the A/B testing framework via HTTP.
Wraps ABTestRunner / ABTester from ab_tester.py and benchmarks queries
against the active psycopg2 session managed by db_router.py.

Endpoints
---------
POST /api/db/test/ab           — run full A/B comparison (original vs optimized)
POST /api/db/test/single       — benchmark a single query (baseline timing)
GET  /api/db/test/results/{id} — retrieve a stored test result by run-id
GET  /api/db/test/results      — list all stored run IDs
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from ab_tester import ABTestRunner, ABTester, ComparisonResult, TestResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db/test", tags=["ab-test"])

# ---------------------------------------------------------------------------
# In-memory result store  (swap for Redis / DB in production)
# ---------------------------------------------------------------------------
_result_store: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Session helpers — share the same params store as db_router.py
# ---------------------------------------------------------------------------

# SESSION_STORE is the public alias for db_router._sessions.
# Both dicts point to the same object in memory, so sessions registered via
# POST /api/db/connect are immediately visible here.
try:
    from routers.db_router import SESSION_STORE  # production import path
except ImportError:
    try:
        from db_router import SESSION_STORE      # flat layout / test fallback
    except ImportError:
        SESSION_STORE: Dict[str, Any] = {}       # unit-test stub


def _get_params(session_id: str) -> dict:
    """
    Return the stored connection params dict for *session_id*, or raise 401.
    Mirrors the pattern used in db_router._get_conn() so every benchmark
    opens its own fresh connection rather than sharing one across threads.
    """
    params = SESSION_STORE.get(session_id)
    if params is None:
        raise HTTPException(
            status_code=401,
            detail="No active session. Connect to a database first.",
        )
    return params


def _require_session_id(x_session_id: str = Header(..., alias="X-Session-ID")) -> str:
    return x_session_id


# ---------------------------------------------------------------------------
# Executor adapter — opens a fresh connection per benchmark run
# ---------------------------------------------------------------------------

class _PsycopgExecutor:
    """
    Bridges ABTester's executor.execute(query) contract to psycopg2.

    A *fresh* connection is opened from the stored params on every call so:
      • benchmark runs are isolated from each other
      • we match the same pattern as db_router._get_conn()
      • there is no shared-connection state to corrupt across iterations

    Every execution is rolled back immediately so benchmarks never mutate data.
    """

    def __init__(self, params: dict):
        self._params = params

    def execute(self, query: str) -> Dict[str, Any]:
        conn = psycopg2.connect(**self._params)
        try:
            conn.set_session(autocommit=False)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                try:
                    rows = cur.fetchall()
                except psycopg2.ProgrammingError:
                    # DML with no result set (INSERT / UPDATE / DELETE)
                    rows = []
            conn.rollback()   # never persist mutations from benchmarks
            return {
                "rows_returned": len(rows),
                "rows": [dict(r) for r in rows[:100]],  # cap payload size
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ABTestRequest(BaseModel):
    original_query:  str = Field(..., description="Baseline SQL query")
    optimized_query: str = Field(..., description="Optimized SQL query to compare")
    iterations:      int = Field(default=5,  ge=1, le=20, description="Runs per query")
    warmup:          int = Field(default=1,  ge=0, le=5,  description="Warmup runs before measurement")
    threshold_pct:   float = Field(default=10.0, ge=0.0, description="Min improvement % to call it significant")
    export_id:       Optional[str] = Field(default=None, description="Optional run-id for later retrieval")


class SingleQueryBenchmarkRequest(BaseModel):
    query:      str = Field(..., description="SQL query to benchmark")
    iterations: int = Field(default=5, ge=1, le=20)
    warmup:     int = Field(default=1, ge=0, le=5)


class TestResultResponse(BaseModel):
    query_name:     str
    iterations:     int
    average_ms:     float
    min_ms:         float
    max_ms:         float
    stddev_ms:      float
    average_rows:   float


class ABTestResponse(BaseModel):
    run_id:            str
    original:          TestResultResponse
    optimized:         TestResultResponse
    improvement_pct:   float
    speedup_factor:    float
    is_faster:         bool
    is_significant:    bool
    verdict:           str
    time_saved_ms:     float


class SingleBenchmarkResponse(BaseModel):
    run_id:        str
    query_name:    str
    iterations:    int
    average_ms:    float
    min_ms:        float
    max_ms:        float
    stddev_ms:     float
    average_rows:  float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_test_result(r: TestResult) -> TestResultResponse:
    return TestResultResponse(
        query_name=r.query_name,
        iterations=r.iterations,
        average_ms=round(r.average_time_ms, 3),
        min_ms=round(r.min_time_ms, 3),
        max_ms=round(r.max_time_ms, 3),
        stddev_ms=round(r.stddev_ms, 3),
        average_rows=r.average_rows,
    )


def _serialize_comparison(
    run_id: str,
    comparison: ComparisonResult,
    threshold: float,
    runner: ABTestRunner,
) -> ABTestResponse:
    is_significant = comparison.improvement_percentage >= threshold
    verdict = runner._generate_verdict(comparison)

    return ABTestResponse(
        run_id=run_id,
        original=_serialize_test_result(comparison.original),
        optimized=_serialize_test_result(comparison.optimized),
        improvement_pct=round(comparison.improvement_percentage, 2),
        speedup_factor=round(comparison.speedup_factor, 3),
        is_faster=comparison.is_faster,
        is_significant=is_significant,
        verdict=verdict,
        time_saved_ms=round(
            comparison.original.average_time_ms - comparison.optimized.average_time_ms, 3
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/ab", response_model=ABTestResponse, summary="Run A/B test: original vs optimized")
def run_ab_test(
    body: ABTestRequest,
    session_id: str = Depends(_require_session_id),
) -> ABTestResponse:
    """
    Execute both queries N times, collect timing metrics, and return a
    structured comparison including improvement %, speedup factor, and verdict.

    The connection is ROLLED BACK after every execution so benchmarks never
    mutate your data.
    """
    params = _get_params(session_id)
    executor = _PsycopgExecutor(params)

    run_id = body.export_id or str(uuid.uuid4())[:8]

    runner = ABTestRunner(
        executor=executor,
        config={
            "iterations":             body.iterations,
            "warmup_iterations":      body.warmup,
            "threshold_improvement_pct": body.threshold_pct,
        },
    )

    try:
        comparison = runner.tester.compare_queries(
            original_query=body.original_query,
            optimized_query=body.optimized_query,
        )
    except Exception as exc:
        logger.exception("A/B test failed for session %s", session_id)
        raise HTTPException(status_code=400, detail=str(exc))

    response = _serialize_comparison(run_id, comparison, body.threshold_pct, runner)

    # Persist for later retrieval (model_dump() for Pydantic v2, dict() for v1)
    try:
        _result_store[run_id] = response.model_dump()
    except AttributeError:
        _result_store[run_id] = response.dict()
    logger.info("A/B run %s complete — improvement: %.1f%%", run_id, response.improvement_pct)

    return response


@router.post("/single", response_model=SingleBenchmarkResponse, summary="Benchmark a single query")
def benchmark_single(
    body: SingleQueryBenchmarkRequest,
    session_id: str = Depends(_require_session_id),
) -> SingleBenchmarkResponse:
    """
    Run one query N times and return timing stats.
    Useful for establishing a baseline before generating an optimized version.
    """
    params = _get_params(session_id)
    executor = _PsycopgExecutor(params)

    tester = ABTester(
        executor=executor,
        iterations=body.iterations,
        warmup_iterations=body.warmup,
    )

    run_id = str(uuid.uuid4())[:8]

    try:
        result = tester.run_test(body.query, query_name="query")
    except Exception as exc:
        logger.exception("Single benchmark failed for session %s", session_id)
        raise HTTPException(status_code=400, detail=str(exc))

    response = SingleBenchmarkResponse(
        run_id=run_id,
        query_name=result.query_name,
        iterations=result.iterations,
        average_ms=round(result.average_time_ms, 3),
        min_ms=round(result.min_time_ms, 3),
        max_ms=round(result.max_time_ms, 3),
        stddev_ms=round(result.stddev_ms, 3),
        average_rows=result.average_rows,
    )

    try:
        _result_store[run_id] = response.model_dump()
    except AttributeError:
        _result_store[run_id] = response.dict()
    return response


@router.get("/results/{run_id}", summary="Retrieve a stored test result")
def get_result(run_id: str) -> Dict[str, Any]:
    """
    Fetch a previously stored A/B or single-benchmark result by its run-id.
    Results are held in memory — they reset on server restart.
    """
    result = _result_store.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result found for run_id '{run_id}'")
    return result


@router.get("/results", summary="List all stored run IDs")
def list_results() -> List[str]:
    """Return all run IDs currently in the result store."""
    return list(_result_store.keys())