# MMAD Inspector (Step 1 / MVP)

**What you get in Step 1**
- ✅ MMAD(`mmad.json`) loader + MMAD-style MCQ evaluation runner
- ✅ End-to-end demo pipeline: **image → anomaly model → structured defect → report(JSON)**
- ✅ FastAPI inference server + Streamlit dashboard
- ✅ Paths are fully configurable via `.env` / `configs/runtime.yaml` so you can move between local ↔ Colab ↔ AWS easily.
- ✅ Designed so Step 2 can add **PatchCore / AnomalyCLIP** without rewriting the app.

> Step 1 intentionally uses **DummyEdgeAnomaly** + **EchoMLLM** so it runs on CPU and the pipeline is testable.
> You will replace these with real models in Step 2.

---

## 0. Folder structure

```
mmad_inspector_mvp/
  apps/
    api/          # FastAPI
    dashboard/    # Streamlit
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

2. Edit `.env`:
- `MMAD_DATA_ROOT`: root folder that contains images like `DS-MVTec/...png`
- `MMAD_JSON_PATH`: path to `mmad.json`

Or directly edit `configs/runtime.yaml`.

---

## 3) Run evaluation (MMAD-style accuracy)

```bash
python scripts/eval_mmad.py --config configs/eval.yaml
```

This will create:
- `outputs/eval/mmad_eval.json` (pred records + summary)

---

## 4) Run API + Dashboard

Terminal A:
```bash
uvicorn apps.api.main:app --reload --port 8000
```

Terminal B:
```bash
streamlit run apps/dashboard/app.py
```

Terminal C (React frontend):
```bash
cd apps/frontend
npm ci
cp .env.example .env.local
npm run dev -- --host 0.0.0.0 --port 5173
```

See deployment/handoff runbook: `docs/deploy-and-handoff.md`

---

## Step 2: how to extend (PatchCore / AnomalyCLIP)
When you add PatchCore/AnomalyCLIP later, you only need to implement `AnomalyModel.infer()` in:
- `src/mmad_inspector/anomaly/patchcore_adapter.py`
- `src/mmad_inspector/anomaly/anomalyclip_adapter.py`

Then switch in `configs/runtime.yaml`:
```yaml
anomaly:
  model: patchcore
```

Everything else (API/UI/eval/report schema) stays the same.
