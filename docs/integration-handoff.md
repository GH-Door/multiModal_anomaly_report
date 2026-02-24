# Integration Handoff (Frontend/Backend Parallel Work)

## Goal
- Keep the current web UI value flow stable while migrating away from `legacy/`.
- Allow teammates to merge unfinished frontend/backend features later with minimal conflicts.

## Canonical Runtime Paths
- API entrypoint: `apps/api/main.py`
- Service layer: `src/service/ad_service.py`, `src/service/llm_service.py`, `src/service/visual_rag_service.py`
- DB layer: `src/storage/pg.py`

Do not add runtime imports from `legacy/*`.

## API Contract to Keep Stable
### `POST /inspect`
- Request form fields:
  - `file` (required)
  - `category` (required)
  - `dataset` (optional, default `default`)
  - `line` (optional, default `line_1`)
- Response fields:
  - `status`
  - `report_id`
  - `ad_decision`

### `GET /reports`
- Query params:
  - `category` (optional)
  - `decision` (optional)
  - `page` (default `1`)
  - `page_size` (default `5000`)
- Response shape:
  - `page`
  - `page_size`
  - `total_count`
  - `items` (report rows)

## Report Fields Expected by Current Frontend
- `dataset`, `category`, `line`
- `ad_score`, `ad_decision`, `is_anomaly_ad`
- `has_defect`, `region`, `area_ratio`, `confidence`, `bbox`, `center`
- `image_path`, `heatmap_path`, `mask_path`, `overlay_path`, `similar_image_path`
- `llm_report`, `llm_summary`, `llm_inference_duration`, `is_anomaly_llm`
- `applied_policy`, `created_at`

## Merge Rules for Teammates
- Frontend PRs: keep API payload parsing backward-compatible.
- Backend PRs: do not remove existing response keys; only additive changes.
- DB changes: additive migrations only (`ADD COLUMN IF NOT EXISTS` style).

## Legacy-to-Repo Copy Map (when teammate debug is done)
- `legacy/apps/Frontend/*` (except node_modules/env/log) -> `apps/frontend/`
- `legacy/apps/app/main.py` changes -> `apps/api/main.py`
- `legacy/apps/app/services/*.py` changes -> `src/service/*.py`
- `legacy/apps/src/storage/*.py` changes -> `src/storage/*.py`

Recommended frontend sync command:
```bash
rsync -a --exclude node_modules --exclude .env.local --exclude frontend.log \
  legacy/apps/Frontend/ apps/frontend/
```

## Open Work (for teammate merge)
- Endpoint and DTO refinements for unfinished UI features.
- Production auth/permission/cors hardening.
- Async/batch inference optimization.
