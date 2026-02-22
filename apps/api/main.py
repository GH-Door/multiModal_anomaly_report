from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import src.storage.pg as pg
from src.mllm.factory import get_llm_client, list_llm_models
from src.service.ad_service import AdService
from src.service.llm_service import LlmService
from src.service.visual_rag_service import RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("PG_DSN", "postgresql://son:1234@localhost/inspection"))
MODEL_NAME = os.getenv("LLM_MODEL", "internvl")
CHECKPOINT_DIR = Path(os.getenv("AD_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "rag_index")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "datasets")))

INCOMING_ROOT = Path(os.getenv("INCOMING_ROOT", "/home/ubuntu/incoming"))
INCOMING_SCAN_INTERVAL_SEC = float(os.getenv("INCOMING_SCAN_INTERVAL_SEC", "5"))
INCOMING_STABLE_SECONDS = float(os.getenv("INCOMING_STABLE_SECONDS", "2"))
INCOMING_DEFAULT_DATASET = os.getenv("INCOMING_DEFAULT_DATASET", "incoming")
INCOMING_DEFAULT_CATEGORY = os.getenv("INCOMING_DEFAULT_CATEGORY", "")
INCOMING_DEFAULT_LINE = os.getenv("INCOMING_DEFAULT_LINE", "LINE-A-01")
INCOMING_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

LINE_PATTERN = re.compile(r"(?i)line[_-]?([a-z0-9]+)")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


INCOMING_WATCH_ENABLED = _env_bool("INCOMING_WATCH_ENABLED", True)


class LlmModelUpdateRequest(BaseModel):
    model: str


class LocalImageUpload:
    """UploadFile-like wrapper for filesystem image ingestion."""

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.file = path.open("rb")

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


class CachedStaticFiles(StaticFiles):
    """Static file mount with cache headers for immutable output filenames."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if getattr(response, "status_code", 0) == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
        return response


def _set_llm_model(app: FastAPI, model_name: str) -> None:
    selected = model_name.strip()
    if not selected:
        raise ValueError("Model name must not be empty")
    app.state.llm_client = get_llm_client(selected)
    app.state.llm_service = LlmService(client=app.state.llm_client)
    app.state.llm_model = selected


def _checkpoint_exists_for_category_key(category_key: str) -> bool:
    key = category_key.strip().strip("/")
    if not key:
        return False

    roots = [CHECKPOINT_DIR / "Patchcore", CHECKPOINT_DIR]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        category_dir = root / key
        if category_dir.exists() and category_dir.is_dir():
            for v_dir in category_dir.iterdir():
                if not v_dir.is_dir() or not v_dir.name.startswith("v"):
                    continue
                if (v_dir / "model.ckpt").exists():
                    return True
    return False


def _resolve_model_category(dataset: str, category: str) -> str:
    selected = category.strip()
    if not selected:
        raise ValueError("Category must not be empty")
    if "/" in selected:
        return selected

    ds = dataset.strip().strip("/")
    if ds:
        candidate = f"{ds}/{selected}"
        if _checkpoint_exists_for_category_key(candidate):
            return candidate
    return selected


def _normalize_line(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    up = s.upper()
    if up.startswith("LINE-"):
        return up

    m = LINE_PATTERN.search(s)
    if not m:
        return s

    token = m.group(1).upper()
    if len(token) == 1 and token.isalpha():
        return f"LINE-{token}-01"
    return f"LINE-{token}"


def _resolve_incoming_context(image_path: Path) -> tuple[str, str, str] | None:
    try:
        rel_parts = image_path.relative_to(INCOMING_ROOT).parts
    except ValueError:
        rel_parts = image_path.parts

    dataset = rel_parts[0] if len(rel_parts) >= 1 else INCOMING_DEFAULT_DATASET
    category = rel_parts[1] if len(rel_parts) >= 2 else INCOMING_DEFAULT_CATEGORY

    line: str | None = None
    # 권장 구조: incoming/{dataset}/{category}/{line}/{batch}/image.png
    if len(rel_parts) >= 3:
        line = _normalize_line(rel_parts[2])

    # line 디렉토리가 없으면 하위 경로(배치명 등)에서 line token 추출
    if line is None and len(rel_parts) >= 3:
        for token in reversed(rel_parts[2:-1]):  # 파일명 제외
            line = _normalize_line(token)
            if line is not None:
                break

    line = line or INCOMING_DEFAULT_LINE

    if not category:
        return None
    return dataset, category, line


def _collect_incoming_images(root: Path) -> list[Path]:
    items: list[tuple[float, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in INCOMING_IMAGE_EXTS:
            continue
        try:
            items.append((p.stat().st_mtime, p))
        except OSError:
            continue
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _is_stable_file(path: Path) -> bool:
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
        return age >= INCOMING_STABLE_SECONDS
    except OSError:
        return False


def _run_inspection_pipeline(
    app: FastAPI,
    *,
    conn,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
) -> dict[str, Any]:
    model_category = _resolve_model_category(dataset=dataset, category=category)
    policy = pg.get_category_policy(conn, model_category)
    t_low = float(policy.get("t_low", 0.5))
    t_high = float(policy.get("t_high", 0.8))
    policy_source = str(policy.get("source", "default"))

    if hasattr(file_obj, "file"):
        file_obj.file.seek(0)

    ad_start_time = datetime.now()
    # GPU/model access is serialized for stability; downstream RAG/LLM stays concurrent.
    with app.state.inspect_lock:
        ad_results = app.state.ad_service.predict_batch(
            [file_obj],
            category=model_category,
            dataset=dataset,
            threshold=t_low,
        )
    ad_result = ad_results[0]
    score = float(ad_result["ad_score"])
    model_thr = float(ad_result.get("model_threshold", t_low))
    model_is_anomaly = bool(ad_result.get("model_is_anomaly", score > model_thr))

    # category metadata가 없으면 run_ad_inference 기본 흐름처럼
    # model threshold 중심의 review band를 사용한다.
    if policy_source != "category_metadata":
        center = max(0.0, min(1.0, model_thr))
        band = 0.08
        t_low = max(0.0, center - band)
        t_high = min(1.0, center + band)

    # 정책 파일이 잘못 들어간 경우(all anomaly/normal 쏠림) 방어.
    if t_high <= t_low or not (0.0 <= t_low <= 1.0 and 0.0 <= t_high <= 1.0):
        logger.warning(
            "Invalid policy band for %s (t_low=%.4f, t_high=%.4f). source=%s fallback to model threshold band.",
            model_category,
            t_low,
            t_high,
            policy_source,
        )
        center = max(0.0, min(1.0, model_thr))
        t_low = max(0.0, center - 0.05)
        t_high = min(1.0, center + 0.05)

    if score < t_low:
        ad_decision = "normal"
    elif score > t_high:
        ad_decision = "anomaly"
    else:
        ad_decision = "review_needed"

    logger.info(
        "AD decision | category=%s score=%.4f band=[%.4f, %.4f] source=%s model_thr=%.4f model_is_anomaly=%s decision=%s",
        model_category,
        score,
        t_low,
        t_high,
        policy_source,
        model_thr,
        model_is_anomaly,
        ad_decision,
    )

    initial_data = {
        "dataset": dataset,
        "category": model_category,
        "line": line,
        "ad_score": score,
        "ad_decision": ad_decision,
        "is_anomaly_ad": score > t_high,
        "has_defect": ad_result.get("has_defect", False),
        "region": ad_result.get("region"),
        "area_ratio": ad_result.get("area_ratio"),
        "bbox": ad_result.get("bbox"),
        "center": ad_result.get("center"),
        "confidence": ad_result.get("confidence"),
        "ingest_source_path": ingest_source_path,
        "image_path": ad_result["original_path"],
        "heatmap_path": ad_result["heatmap_path"],
        "mask_path": ad_result["mask_path"],
        "overlay_path": ad_result["overlay_path"],
        "created_at": ad_start_time,
        "ad_inference_duration": (datetime.now() - ad_start_time).total_seconds(),
        "applied_policy": policy,
    }
    report_id = pg.insert_report(conn, initial_data)

    warnings: list[str] = []
    ref_path = None
    try:
        rag_results = app.state.rag_service.search_closest_normal(ad_result["original_path"], model_category)
        ref_path = rag_results[0]["path"] if rag_results else None
        pg.update_report(conn, report_id, {"similar_image_path": ref_path})
    except Exception:
        logger.exception("RAG stage failed for report_id=%s", report_id)
        warnings.append("rag_failed")

    try:
        llm_response = app.state.llm_service.generate_dynamic_report(
            ad_result["original_path"],
            ref_path,
            model_category,
            {**ad_result, "ad_decision": ad_decision},
            policy,
        )
        pg.update_report(conn, report_id, llm_response)
    except Exception:
        logger.exception("LLM stage failed for report_id=%s", report_id)
        warnings.append("llm_failed")

    return {
        "status": "success",
        "report_id": report_id,
        "ad_decision": ad_decision,
        "category": model_category,
        "source_image": ingest_source_path,
        "warnings": warnings,
    }


def _run_single_inspection(
    app: FastAPI,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
) -> dict[str, Any]:
    conn = pg.connect_fast(DATABASE_URL)
    try:
        return _run_inspection_pipeline(
            app,
            conn=conn,
            file_obj=file_obj,
            category=category,
            dataset=dataset,
            line=line,
            ingest_source_path=ingest_source_path,
        )
    finally:
        conn.close()


def _scan_incoming_once(app: FastAPI) -> int:
    if not INCOMING_ROOT.exists():
        return 0

    conn = pg.connect_fast(DATABASE_URL)
    processed = 0
    try:
        for image_path in _collect_incoming_images(INCOMING_ROOT):
            if not _is_stable_file(image_path):
                continue

            source_path = str(image_path.resolve())
            if pg.has_ingest_source_path(conn, source_path):
                continue

            resolved = _resolve_incoming_context(image_path)
            if resolved is None:
                unresolved: set[str] = app.state.incoming_unresolved
                if source_path not in unresolved:
                    logger.warning(
                        "Skipping incoming image without category mapping: %s (use incoming/{dataset}/{category}/... structure or set INCOMING_DEFAULT_CATEGORY)",
                        source_path,
                    )
                    unresolved.add(source_path)
                continue
            dataset, category, line = resolved

            local_file = LocalImageUpload(image_path)
            try:
                _run_inspection_pipeline(
                    app,
                    conn=conn,
                    file_obj=local_file,
                    category=category,
                    dataset=dataset,
                    line=line,
                    ingest_source_path=source_path,
                )
                processed += 1
                app.state.incoming_unresolved.discard(source_path)
                logger.info("Incoming image processed: %s", source_path)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logger.exception("Incoming image processing failed: %s", source_path)
            finally:
                local_file.close()

        return processed
    finally:
        conn.close()


async def _incoming_watch_loop(app: FastAPI) -> None:
    logger.info(
        "Incoming watcher enabled | root=%s | interval=%.1fs | stable=%.1fs",
        INCOMING_ROOT,
        INCOMING_SCAN_INTERVAL_SEC,
        INCOMING_STABLE_SECONDS,
    )
    while True:
        try:
            processed = await run_in_threadpool(_scan_incoming_once, app)
            if processed > 0:
                logger.info("Incoming watcher processed %d image(s)", processed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Incoming watcher loop failed")

        await asyncio.sleep(max(1.0, INCOMING_SCAN_INTERVAL_SEC))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot-time DB connectivity and additive migrations
    bootstrap_conn = pg.connect(DATABASE_URL, ensure_schema=True)
    bootstrap_conn.close()

    app.state.ad_service = AdService(
        checkpoint_dir=str(CHECKPOINT_DIR),
        output_root=str(OUTPUT_DIR),
    )
    app.state.rag_service = RagService(index_dir=str(RAG_INDEX_DIR))
    _set_llm_model(app, MODEL_NAME)
    app.state.inspect_lock = threading.Lock()
    app.state.incoming_unresolved = set()

    incoming_task: asyncio.Task | None = None
    if INCOMING_WATCH_ENABLED:
        INCOMING_ROOT.mkdir(parents=True, exist_ok=True)
        incoming_task = asyncio.create_task(_incoming_watch_loop(app))

    try:
        yield
    finally:
        if incoming_task is not None:
            incoming_task.cancel()
            try:
                await incoming_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Industrial AI Inspection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", CachedStaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="outputs")
app.mount("/data/datasets", StaticFiles(directory=str(DATA_DIR), check_dir=False), name="datasets")


@app.get("/settings/llm-model")
async def get_llm_model_settings():
    return {
        "active_model": str(getattr(app.state, "llm_model", MODEL_NAME)),
        "available_models": list_llm_models(),
    }


@app.put("/settings/llm-model")
async def update_llm_model_settings(payload: LlmModelUpdateRequest):
    model_name = payload.model.strip()
    try:
        _set_llm_model(app, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to switch LLM model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "active_model": str(app.state.llm_model),
        "available_models": list_llm_models(),
    }


@app.get("/incoming/status")
async def get_incoming_status():
    return {
        "enabled": INCOMING_WATCH_ENABLED,
        "root": str(INCOMING_ROOT),
        "scan_interval_sec": INCOMING_SCAN_INTERVAL_SEC,
        "stable_seconds": INCOMING_STABLE_SECONDS,
        "default_dataset": INCOMING_DEFAULT_DATASET,
        "default_category": INCOMING_DEFAULT_CATEGORY,
        "default_line": INCOMING_DEFAULT_LINE,
    }


@app.post("/inspect")
async def run_inspection(
    file: UploadFile = File(...),
    category: str = Form(...),
    dataset: str = Form("default"),
    line: str = Form("line_1"),
):
    """Run AD -> RAG -> LLM and persist report fields used by current frontend."""
    try:
        file.file.seek(0)
        return await run_in_threadpool(
            _run_single_inspection,
            app,
            file,
            category,
            dataset,
            line,
            None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inspection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports")
async def get_reports(
    category: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=5000),
    limit: Optional[int] = Query(None, ge=1, le=5000),
    offset: Optional[int] = Query(None, ge=0),
    include_full: bool = Query(False),
):
    """Fetch reports in the same shape expected by current frontend."""
    if limit is not None or offset is not None:
        effective_limit = int(limit or page_size)
        effective_offset = int(offset or 0)
        effective_page = (effective_offset // max(1, effective_limit)) + 1
    else:
        effective_limit = int(page_size)
        effective_offset = (page - 1) * page_size
        effective_page = int(page)

    conn = pg.connect_fast(DATABASE_URL)
    try:
        total = pg.count_filtered_reports(
            conn,
            category=category,
            decision=decision,
        )
        reports = pg.get_filtered_reports(
            conn,
            category=category,
            decision=decision,
            limit=effective_limit,
            offset=effective_offset,
            include_full=include_full,
        )
    except Exception as exc:
        logger.exception("Report list fetch failed")
        raise HTTPException(status_code=500, detail="데이터 조회 중 오류가 발생했습니다.") from exc
    finally:
        conn.close()

    return {
        "page": effective_page,
        "page_size": effective_limit,
        "offset": effective_offset,
        "total_count": total,
        "total": total,
        "items": reports,
    }
