# NeuroScan Architecture

## High-Level Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Scan Upload │────▶│ load_medical │────▶│ RL Preprocessor │
│ (PNG/DICOM) │     │ _image()     │     │ (Q-table agent) │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                     ┌─────────────────────────────┘
                     ▼
              ┌──────────────┐     ┌──────────────────┐
              │ Generalist   │────▶│ Specialist CNN   │
              │ ResNet-50    │     │ (per category)   │
              └──────────────┘     └────────┬─────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────┐
         ▼                                  ▼                      ▼
  ┌─────────────┐                  ┌──────────────┐        ┌────────────┐
  │ Confidence  │                  │ RAG Pipeline │        │ MongoDB    │
  │ < 90% flag  │                  │ (Gemini +    │        │ agent logs │
  └─────────────┘                  │  vector DB)  │        └────────────┘
                                   └──────────────┘
```

## Application Layers (Current: Single File)

All logic currently lives in `app.py`. The sections below map to numbered comment blocks in that file.

### 1. Bootstrap & Optional Dependencies (lines 1–83)

- UTF-8 stdout/stderr fix for Windows
- Graceful imports: `dotenv`, `pydicom`, `boto3`, `sentence_transformers`, `google.generativeai`
- Matplotlib `Agg` backend for headless plotting

### 2. Neural Network Architectures (lines 88–293)

| Class | Purpose | Output classes |
|-------|---------|----------------|
| `DSConv`, `LiteResBlock2` | Building blocks | — |
| `LiteBrainNet2` | Genetic / malformations brain specialist | 3–4 |
| `InfectiousBrainNet` | Infectious brain diseases | 5 |
| `StableMetabolicNet` | Metabolic disorders | 2 |
| `NeoplasticBrainNet` | Brain tumors | 2 |
| `CustomLiverNet` | Malignant liver lesions | 3 |
| `MicroLiverNet` | Ductal liver conditions | 2 |

### 3. MedicalAIHub (lines 298–405)

Central inference orchestrator:

- Loads ResNet-50 generalist weights
- `get_specialist_instance(category)` — maps category string → architecture + weight path + label list
- `diagnose_array(img_np)` — generalist → confidence gate (90%) → specialist
- Image transform: resize 224×224, ImageNet normalization

### 4. RL Preprocessing (lines 410–541)

**State representation:** 4 metrics discretized to integers (brightness, contrast, noise, edge density).

**Actions:** `clahe`, `median`, `gaussian`, `sharpen`, `stop`, `none`

**Policy:**
1. If initial generalist confidence ≥ 90% → skip preprocessing
2. Otherwise, up to 2 steps using Q-table lookup
3. Unknown states → greedy fallback (try all transforms, pick best confidence boost)
4. Early stop when confidence ≥ 90%

**Training artifact:** `rl_agent.json` contains `Q` dict mapping state keys → action Q-values.

### 5. Persistence Layer (lines 546–721)

| Function | Role |
|----------|------|
| `get_mongodb_connection()` | Connect to `NeuroScan_DB` (duplicated definition — second wins) |
| `upload_to_s3()` | Optional scan archival |
| `load_medical_image()` | PNG/JPEG via OpenCV; DICOM via pydicom |
| `log_agent_draft()` | Insert analysis record to `agent_result_logs` |
| `submit_doctor_feedback()` | Update HITL feedback on existing document |

**MongoDB document schema (draft):**
```json
{
  "timestamp": "UTC datetime",
  "execution_status": "Successful Run | Low Confidence Flagged | ...",
  "patient_name": "string",
  "patient_id": "string",
  "s3_url": "string | null",
  "agent_steps": ["clahe", "gaussian", ...],
  "diagnostic_context": { "disease": "...", "confidence": 0.94 },
  "human_in_the_loop": { "status": "Pending UI Feedback | Feedback Submitted", ... }
}
```

### 6. LightVectorDB (lines 726–795)

In-memory dual-collection vector store:

- `medical_base` — JSON-serialized disease profiles from `DISEASE_DB`
- `agent_result_logs` — synthetic historical case narratives

Embedding via SentenceTransformer; cosine similarity search. Falls back to keyword overlap if embedder unavailable.

### 7. MedicalRAGPipeline (lines 800–931)

1. Query both collections (top-k=2 each)
2. Build structured prompt with textbook + historical context
3. Generate via Gemini (model auto-selection) or rule-engine fallback

### 8. Clinical Knowledge (lines 937–1331)

- `DISEASE_DB` — 20 disease entries with MRI findings, clinical features, prognosis, references
- `seed_vector_database()` — populates vector DB at startup

### 9. Gradio UI & Orchestration (lines 1337–1623)

**Global singletons initialized at import:**
- `hub`, `vector_db`, `rag_pipeline`, `db_client`, `Q_table`

**Main handlers:**
- `analyze_scan()` — full pipeline: load → S3 → RL → classify → RAG → MongoDB log
- `reset_workspace()` — clear UI state
- `submit_doctor_feedback()` — HITL updates

**UI panels:** Connection status, patient metadata, scan upload, before/after images, RL logs, hierarchical diagnosis, S3 URL, doctor feedback, RAG report.

## External Integrations

```
┌────────────┐   ┌───────────────┐   ┌─────────────┐
│ Gradio UI  │──▶│ app.py core   │──▶│ PyTorch GPU │
└────────────┘   └───────┬───────┘   └─────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   MongoDB Atlas    AWS S3          Gemini API
```

## Known Technical Debt

1. **Duplicate `get_mongodb_connection()`** — First definition (localhost default) is shadowed by second (Atlas credentials hardcoded)
2. **Hardcoded MongoDB credentials** — Should be env-only
3. **Global mutable state** — `db_client`, singleton hub at module level
4. **Monolithic file** — ~1,620 lines; see refactor plan
5. **DICOM fallback** — Returns random array on failure (demo behavior)
6. **Flask in requirements** — Not used in `app.py`

## Confidence Thresholds

| Stage | Threshold | Behavior |
|-------|-----------|----------|
| RL skip | ≥ 90% initial | No preprocessing |
| RL stop | ≥ 90% post-step | End preprocessing loop |
| Specialist gate | < 90% generalist | Return "Low Confidence" alert, skip specialist RAG |
