# NeuroScan — Adaptive Medical Image Diagnostic Station

A full-stack medical imaging workstation that combines hierarchical deep learning, reinforcement-learning-based preprocessing, retrieval-augmented generation (RAG), and human-in-the-loop feedback logging.

## Overview

**NeuroScan** (branded in the UI as *NeuroScan Workstation*) is a Gradio web application for analyzing brain and liver MRI/CT scan slices. The system uses a two-stage classifier:

1. **Generalist (Level 1)** — ResNet-50 routes the scan into a broad clinical category (genetic, infectious, metabolic, neoplastic, malignant, ductal, etc.).
2. **Specialist (Level 2)** — Category-specific CNN architectures produce a fine-grained pathology label.

When the generalist confidence is below 90%, an **RL preprocessing agent** applies image enhancements (CLAHE, median blur, Gaussian blur, sharpen) guided by a pre-trained Q-table to boost classification confidence before specialist routing.

High-confidence diagnoses trigger a **clinical RAG pipeline** that synthesizes patient-facing explanations from textbook disease profiles and historical agent logs, optionally powered by Google Gemini.

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Gradio 3.x |
| ML / CV | PyTorch, torchvision, OpenCV, NumPy |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Google Generative AI (Gemini) |
| Database | MongoDB Atlas (`NeuroScan_DB`) |
| Cloud storage | AWS S3 (optional, presigned URLs) |
| Medical imaging | pydicom (DICOM support) |

## Repository Structure

```
Medical-AI/
├── app.py                      # Monolithic application
├── requirements.txt            # Local dev dependencies (GPU or CPU)
├── requirements-railway.txt    # CPU-only deps for Railway Docker build
├── Dockerfile                  # Railway deployment image
├── railway.toml                # Railway service config
├── .env.example                # Environment variable template
├── models/                     # Optional local model overrides
├── specialist_models/          # Specialist CNN weights (in repo)
├── rl_agent_weights (1).pth    # Generalist ResNet-50 weights (in repo)
├── rl_training_metadata (1).json  # RL Q-table (in repo)
├── docs/
│   ├── ARCHITECTURE.md
│   └── APP_PY_REFACTOR_PLAN.md
└── README.md
```

> **Note:** Model weights live in `specialist_models/` and repo root. Optional copies can go in `models/`.

## Local Model Testing

1. Copy env template:
   ```bash
   cp .env.example .env
   ```
2. Install deps and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
3. On startup, check the console for `✓` / `✗` next to each model file.

Weights are already in the repo — no manual copy step required.

## Railway Deployment (Minimal)

1. Push repo to GitHub (models are **not** in git — see options below).
2. Create a Railway project → **Deploy from GitHub** → select this repo.
3. Set service **memory to at least 2 GB** (4 GB recommended).
4. Add environment variables in Railway dashboard (see `.env.example`).
5. Railway auto-sets `PORT`; the Dockerfile uses CPU-only PyTorch.

**Model weights on Railway:** included in the Docker image via `specialist_models/` and root weight files. Set service **memory to at least 2 GB** (4 GB recommended).

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `MODEL_DIR` | For local testing | Path to model weights (default search: `./models`) |
| `MONGO_URI` | Optional | MongoDB Atlas connection string |
| `GEMINI_API_KEY` | Optional | Gemini RAG explanations |
| `AWS_S3_BUCKET` | Optional | Scan upload bucket |
| `AWS_ACCESS_KEY_ID` | Optional | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | Optional | AWS credentials |
| `AWS_DEFAULT_REGION` | Optional | AWS region (default: `us-east-1`) |
| `PORT` | Auto on Railway | HTTP port (Railway sets this) |
| `RL_QTABLE_PATH` | Optional | Override Q-table file path |

See [.env.example](.env.example) for a copy-paste template.

## Model Assets

| Key | File location |
|-----|---------------|
| `generalist` | `rl_agent_weights (1).pth` (also accepts `rlagent.wt` / `rlagent.pth`) |
| Specialists | `specialist_models/*.pth` |
| Q-table | `rl_training_metadata (1).json` |

## Core Capabilities

- **Multi-format scan upload** — JPEG, PNG, DICOM (`.dcm`)
- **RL-guided preprocessing** — Q-table lookup with fallback heuristic when state is out-of-distribution
- **12 broad categories → 20+ specific pathologies** — Brain (genetic, infectious, malformations, metabolic, neoplastic) and liver (malignant, ductal, benign)
- **Clinical knowledge base** — 20 structured disease profiles (MRI findings, clinical features, prognosis, references)
- **Vector search** — Dual-collection in-memory index (`medical_base`, `agent_result_logs`)
- **MongoDB logging** — Agent runs, confidence, preprocessing steps, S3 URLs, doctor feedback
- **Human-in-the-loop** — Radiologists can submit corrections linked to analysis sessions

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Pipeline stages, class diagram, and integration points
- [app.py Refactor Plan](docs/APP_PY_REFACTOR_PLAN.md) — Proposed module breakdown and migration phases

## Branch

Documentation was created on branch `temp` (local only; not pushed to GitHub).

## Disclaimer

This software is for research and demonstration purposes. It is **not** a certified medical device. All outputs require review by qualified clinical personnel before any clinical use.
