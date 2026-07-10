# app.py Modularization Plan

This document outlines a phased plan to break the monolithic `app.py` (~1,620 lines) into a maintainable Python package. No code changes are included here — this is the blueprint only.

## Goals

1. **Separation of concerns** — ML, storage, RAG, and UI in distinct modules
2. **Testability** — Unit-test inference, RL, and RAG without launching Gradio
3. **Safer configuration** — Remove hardcoded credentials; centralize env loading
4. **Incremental migration** — Each phase leaves the app runnable

## Target Package Structure

```
medical_ai/
├── __init__.py
├── main.py                     # Entry point: python -m medical_ai.main
├── config/
│   ├── __init__.py
│   ├── settings.py             # Env vars, paths, thresholds
│   └── model_paths.py          # Weight file resolution (find_file)
├── models/
│   ├── __init__.py
│   ├── architectures/
│   │   ├── __init__.py
│   │   ├── blocks.py           # DSConv, LiteResBlock2
│   │   ├── brain.py            # LiteBrainNet2, InfectiousBrainNet, etc.
│   │   └── liver.py            # CustomLiverNet, MicroLiverNet
│   └── hub.py                  # MedicalAIHub class
├── preprocessing/
│   ├── __init__.py
│   ├── transforms.py           # apply_clahe, blur, sharpen, FUNCTION_MAP
│   ├── metrics.py              # analyze_image
│   └── rl_agent.py             # process_image_with_agent_and_hub, Q-table loader
├── storage/
│   ├── __init__.py
│   ├── mongodb.py              # get_mongodb_connection, log_agent_draft, feedback
│   ├── s3.py                   # upload_to_s3
│   └── image_loader.py         # load_medical_image (DICOM + standard)
├── rag/
│   ├── __init__.py
│   ├── vector_db.py            # LightVectorDB
│   ├── pipeline.py             # MedicalRAGPipeline
│   └── knowledge/
│       ├── __init__.py
│       ├── disease_db.py       # DISEASE_DB constant
│       └── seed.py             # seed_vector_database
├── pipeline/
│   ├── __init__.py
│   └── analyze.py              # analyze_scan orchestration
└── ui/
    ├── __init__.py
    ├── gradio_app.py           # Blocks, layout, event wiring
    ├── handlers.py             # reset_workspace, thin wrappers
    └── theme.py                # custom_theme, launch kwargs
```

## Module Mapping (app.py → package)

| app.py lines | Current content | Target module |
|--------------|-----------------|---------------|
| 1–33 | UTF-8 bootstrap | `main.py` or `config/bootstrap.py` |
| 35–83 | Imports + optional deps | `config/settings.py` (flags) |
| 88–293 | CNN architectures | `models/architectures/*.py` |
| 298–405 | `MedicalAIHub` | `models/hub.py` |
| 410–433 | Image transforms | `preprocessing/transforms.py` |
| 435–445 | `analyze_image` | `preprocessing/metrics.py` |
| 447–541 | RL agent loop | `preprocessing/rl_agent.py` |
| 546–555 | MongoDB (1st, duplicate) | **Delete** — keep one impl |
| 557–594 | S3 upload | `storage/s3.py` |
| 596–631 | Image loading | `storage/image_loader.py` |
| 633–649 | MongoDB (2nd) | `storage/mongodb.py` |
| 650–721 | Logging + feedback | `storage/mongodb.py` |
| 726–795 | `LightVectorDB` | `rag/vector_db.py` |
| 800–931 | `MedicalRAGPipeline` | `rag/pipeline.py` |
| 937–1308 | `DISEASE_DB` | `rag/knowledge/disease_db.py` |
| 1310–1331 | `seed_vector_database` | `rag/knowledge/seed.py` |
| 1337–1348 | `find_file` | `config/model_paths.py` |
| 1350–1389 | Global init | `main.py` or `pipeline/factory.py` |
| 1391–1473 | `analyze_scan` | `pipeline/analyze.py` |
| 1475–1476 | `reset_workspace` | `ui/handlers.py` |
| 1478–1623 | Gradio UI + launch | `ui/gradio_app.py`, `main.py` |

## Phased Migration

### Phase 0 — Prep (no behavior change)

- [ ] Create `medical_ai/` package skeleton with empty `__init__.py` files
- [ ] Add `config/settings.py` with `Settings` dataclass reading from `os.environ`
- [ ] Remove duplicate `get_mongodb_connection()`; use env-only `MONGO_URI`
- [ ] Move hardcoded Atlas credentials to `.env.example` (document only)

**Risk:** Low. Config extraction only.

### Phase 1 — Pure functions & data (no torch/gradio imports in new modules yet)

- [ ] Extract `DISEASE_DB` → `rag/knowledge/disease_db.py`
- [ ] Extract `apply_*`, `FUNCTION_MAP`, `analyze_image` → `preprocessing/`
- [ ] Extract `load_medical_image`, `upload_to_s3` → `storage/`
- [ ] Extract `find_file`, path dict builder → `config/model_paths.py`
- [ ] `app.py` imports from new modules (re-export for compatibility)

**Risk:** Low. No class hierarchies moved yet.

### Phase 2 — Model layer

- [ ] Split architectures into `models/architectures/brain.py` and `liver.py`
- [ ] Move `MedicalAIHub` → `models/hub.py`
- [ ] Add `models/__init__.py` factory: `create_hub(paths, classes) -> MedicalAIHub`
- [ ] Unit tests: specialist routing per category string

**Risk:** Medium. Verify weight loading paths unchanged.

### Phase 3 — RL agent

- [ ] Move Q-table loading to `preprocessing/rl_agent.py`
- [ ] Move `process_image_with_agent_and_hub` (inject `hub` dependency)
- [ ] Accept `Q_table` and `max_steps` via config/settings
- [ ] Optional: load Q-table from `rl_training_metadata (1).json` naming convention

**Risk:** Medium. RL loop tightly coupled to `hub._tensor_from_np`.

### Phase 4 — RAG & vector DB

- [ ] Move `LightVectorDB` → `rag/vector_db.py`
- [ ] Move `MedicalRAGPipeline` → `rag/pipeline.py`
- [ ] Move `seed_vector_database` → `rag/knowledge/seed.py`
- [ ] Lazy-init embedder (defer SentenceTransformer load until first query)

**Risk:** Low–medium. Gemini model selection logic stays encapsulated.

### Phase 5 — Storage & pipeline

- [ ] Consolidate MongoDB functions in `storage/mongodb.py`
- [ ] Create `pipeline/analyze.py` with `analyze_scan()` — inject hub, rag, db, Q_table
- [ ] Replace `global db_client` with session/state object or Gradio `gr.State`

**Risk:** Medium. Global state removal requires careful Gradio wiring.

### Phase 6 — UI extraction

- [ ] Move Gradio Blocks to `ui/gradio_app.py` → `build_app(deps) -> gr.Blocks`
- [ ] `main.py`:
  ```python
  def main():
      deps = bootstrap()  # hub, rag, db, Q_table
      demo = build_app(deps)
      demo.launch(...)
  ```
- [ ] Slim `app.py` to:
  ```python
  from medical_ai.main import main
  if __name__ == "__main__":
      main()
  ```
  Or deprecate `app.py` after one release.

**Risk:** Low once pipeline is injectable.

### Phase 7 — Cleanup & hardening

- [ ] Remove unused Flask deps or add separate API module if needed
- [ ] Add `pyproject.toml` / `setup.py` for installable package
- [ ] Add `.env.example` with all required keys
- [ ] Rename `rl_training_metadata (1).json` → `rl_agent.json` or document both
- [ ] Type hints on public APIs
- [ ] Basic pytest suite for preprocessing metrics and hub routing

## Dependency Graph (import order)

```
config/settings.py
    ↓
config/model_paths.py
    ↓
models/architectures/*  →  models/hub.py
    ↓
preprocessing/*  (uses hub interface)
    ↓
storage/*
    ↓
rag/knowledge/*  →  rag/vector_db.py  →  rag/pipeline.py
    ↓
pipeline/analyze.py
    ↓
ui/gradio_app.py  →  main.py
```

**Rule:** `ui/` must not be imported by `models/`, `preprocessing/`, `rag/`, or `storage/`.

## Interface Contracts

### `MedicalAIHub` (keep public API stable)

```python
class MedicalAIHub:
    def diagnose_array(self, img_np: np.ndarray) -> tuple[str, float]: ...
    def get_specialist_instance(self, category: str) -> tuple[nn.Module, str, list[str]]: ...
```

### `analyze_scan` (pipeline entry)

```python
def analyze_scan(
    input_file,
    patient_name: str,
    patient_id: str,
    *,
    hub: MedicalAIHub,
    q_table: dict,
    rag: MedicalRAGPipeline,
    db,
) -> AnalyzeResult:  # named tuple or dataclass
    ...
```

### `build_app` (UI entry)

```python
def build_app(deps: AppDependencies) -> gr.Blocks:
    ...
```

## Testing Strategy (post-refactor)

| Module | Test focus |
|--------|------------|
| `preprocessing/metrics.py` | Deterministic metric output on fixture image |
| `preprocessing/rl_agent.py` | Q-table lookup, fallback action selection |
| `models/hub.py` | Category → specialist mapping (mock weights) |
| `rag/vector_db.py` | Mock embedder keyword fallback |
| `pipeline/analyze.py` | End-to-end with mocked hub/rag/db |

## Estimated Effort

| Phase | Effort | Can ship independently? |
|-------|--------|-------------------------|
| 0 | 2–4 hours | Yes |
| 1 | 4–6 hours | Yes |
| 2 | 4–6 hours | Yes |
| 3 | 3–4 hours | Yes |
| 4 | 3–4 hours | Yes |
| 5 | 4–6 hours | Yes |
| 6 | 3–4 hours | Yes |
| 7 | 4–8 hours | Yes |

**Total:** ~27–42 hours for full modularization.

## What NOT to split (yet)

- **DISEASE_DB content** — Data file, not logic; could later move to JSON/YAML
- **Gradio theme** — Small; keep in `ui/theme.py` only if it grows
- **Kaggle path walking** — Keep in `model_paths.py`; consider env override `MODEL_SEARCH_ROOT`

## Success Criteria

1. `app.py` reduced to &lt; 50 lines (or removed)
2. No duplicate function definitions
3. No hardcoded secrets in source
4. `pytest` passes without GPU (mocked torch where needed)
5. Gradio UI behavior unchanged from user perspective
6. `python -m medical_ai.main` launches identical workstation

## Recommended First PR (smallest valuable slice)

**PR 1:** Phase 0 + Phase 1 + duplicate MongoDB fix

- Creates package skeleton
- Extracts `DISEASE_DB`, preprocessing transforms, image loader
- `app.py` becomes import-and-re-export layer
- ~200 lines moved, zero UI changes

This establishes conventions before touching PyTorch classes or Gradio.
