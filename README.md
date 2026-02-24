# MMAD Inspector (Step 1 / MVP)

**What you get in Step 1**
- ✅ MMAD(`mmad.json`) loader + MMAD-style MCQ evaluation runner
- ✅ End-to-end demo pipeline: **image → anomaly model → structured defect → report(JSON)**
- ✅ FastAPI inference server + React frontend
- ✅ Paths are fully configurable via environment variables
- ✅ Designed so Step 2 can add **PatchCore / AnomalyCLIP** without rewriting the app.

---

## 0. Folder structure

```
mmad_inspector_mvp/
  apps/
    api/          # FastAPI
    frontend/     # React
  configs/
  scripts/
  src/mmad_inspector/
```

---

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2) Configure dataset paths

1. Copy env file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your runtime paths/keys.

---

## 3) Run evaluation (MMAD-style accuracy)

```bash
python scripts/eval_llm_baseline.py --model gemini --few-shot 1 --similar-template
```

---

## 4) Run API + Frontend

Terminal A:
```bash
uvicorn apps.api.main:app --reload --port 8000
```

Terminal B (React frontend):
```bash
cd apps/frontend
npm ci
cp .env.example .env.local
npm run dev -- --host 0.0.0.0 --port 5173
```

See deployment/handoff runbook: `docs/deploy-and-handoff.md`

---

## Step 2: how to extend (PatchCore / AnomalyCLIP)
Everything else (API/UI/eval/report pipeline) stays the same.
