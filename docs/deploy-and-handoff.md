# Deploy and Handoff Runbook

## 1) Push to GitHub
Run from local repo root:

```bash
git status
git add .
git commit -m "[feat]: migrate legacy app into repo apps/src structure"
git push origin <your-branch>
```

Open PR and merge to main (or your target branch).

## 2) Server update (clone first time)
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd multimodal-anomaly-report-generation
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3) Server update (already cloned)
```bash
cd multimodal-anomaly-report-generation
git fetch origin
git checkout main
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Run backend API
```bash
source .venv/bin/activate
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Required env examples:
- `DATABASE_URL`
- `AD_CHECKPOINT_DIR`
- `RAG_INDEX_DIR`
- `OUTPUT_DIR`
- `DATA_DIR`

## 5) Run frontend (React)
```bash
cd apps/frontend
npm ci
cp .env.example .env.local
npm run dev -- --host 0.0.0.0 --port 5173
```

## 6) Optional dashboard
```bash
source .venv/bin/activate
streamlit run apps/dashboard/app.py --server.port 8501
```

## 7) Teammate handoff (legacy-based updates later)
- Keep `legacy/` as reference only (already gitignored).
- Teammates debug on their side, then copy only changed source files into canonical paths:
  - frontend -> `apps/frontend`
  - backend API -> `apps/api`
  - shared Python services -> `src/service`, `src/storage`, `src/mllm`, `src/rag`
- Do not copy generated files (`node_modules`, logs, cache files, `.env.local`).
- Keep API response keys backward-compatible when merging unfinished features.
