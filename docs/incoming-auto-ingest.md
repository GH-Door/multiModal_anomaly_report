# Incoming Auto Ingest Guide

## Goal
Drop production images into an incoming folder, then let the API server auto-run AD+RAG+LLM and save results to `inspection_reports`.

`POST /inspect` remains available for manual debug input.

## Folder-Only Mapping (No meta.json)

The watcher now resolves dataset/category from folder path only.

## Recommended Structure A (Explicit Line Folder)

```text
/home/ubuntu/incoming/
  MVTec-LOCO/
    juice_bottle/
      LINE-A-01/
        260222_18_03/
          image_0001.png
          image_0002.png
```

Mapping rule:
- `dataset` = `MVTec-LOCO`
- `category` = `juice_bottle` (server auto-resolves checkpoint key as `MVTec-LOCO/juice_bottle` when available)
- `line` = `LINE-A-01`

## Recommended Structure B (Line in Batch Name)

```text
/home/ubuntu/incoming/
  MVTec-LOCO/
    juice_bottle/
      260222_18_03_lineA/
        frame_0001.png
        frame_0002.png
```

Line parser examples:
- `lineA` -> `LINE-A-01`
- `line-b-02` -> `LINE-B-02`
- if no line token is found, `INCOMING_DEFAULT_LINE` is used.

## API Env Vars

- `INCOMING_WATCH_ENABLED` (default: `1`)
- `INCOMING_ROOT` (default: `/home/ubuntu/incoming`)
- `INCOMING_SCAN_INTERVAL_SEC` (default: `5`)
- `INCOMING_STABLE_SECONDS` (default: `2`)
- `INCOMING_DEFAULT_DATASET` (default: `incoming`)
- `INCOMING_DEFAULT_CATEGORY` (default: empty; only used when category folder is missing)
- `INCOMING_DEFAULT_LINE` (default: `LINE-A-01`)

## Runtime Check

```bash
curl -s http://127.0.0.1:8000/incoming/status
```

## Notes

- Files are not moved/deleted by watcher.
- Duplicate re-processing is blocked by `ingest_source_path` stored in DB.
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`.
