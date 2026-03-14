# PhysicsFlow — AI-Native Reservoir Simulation & History Matching Platform

> Physics-Informed Neural Operator · Adaptive Ensemble Kalman Inversion ·
> Hybrid RAG Knowledge Assistant · Reservoir Knowledge Graph

**Current version: v2.0.5** — Released 2026-03-14

---

## Changelog

### v2.0.5 (2026-03-14) — AI direct-answer bypass (eliminates hallucination)

**Engine / AI Assistant**

- **`_try_direct_answer()`** — added LLM bypass for well-defined data queries (well performance, production profiles, above/below-expectation groupings, history matching status, mismatch breakdown); the method keyword-matches the user message, calls the relevant tools directly, builds a formatted markdown answer, and streams it without ever invoking Ollama; eliminates hallucinated UI-navigation responses from small models (phi3:mini, etc.) that ignore injected context
- **`_format_well_perf_section()`** — formats above/below-expectation well groups with peak WOPR, water cut, cumulative oil, and per-well RMSE for every well
- **`_format_hm_section()`** — formats HM status: not-started path reports baseline RMSE + well lists; in-progress/converged path reports iteration count, mismatch improvement, and convergence flag
- **`_stream_text()`** — shared word-by-word streamer used by the direct-answer path, yielding the same dict schema as the LLM path for seamless UI compatibility
- Direct answers are saved to conversation history, so follow-up questions still have context

**Desktop** — no binary change; shortcut remains `win-x64-v9`

---

### v2.0.4 (2026-03-14) — AI concise answers & HM status grounding

**Engine / AI Assistant**

- **`get_hm_iteration_summary()`** — no longer returns a bare error when no HM has run; now returns a structured `hm_status="not_started"` dict with `baseline_rmse`, `wells_above_expectation`, `wells_below_expectation`, and a single actionable note so the model always has numbers to quote
- **`get_project_summary()`** — HM state now inlined in every system context block: when HM has run → iteration count + initial/final mismatch + % improvement + convergence flag; when not run → baseline RMSE + "Start αREKI" note
- **System prompt rules 5-7** added: model must quote RMSE and well lists when `hm_status="not_started"`; one-sentence-only response when any tool errors; model may use Active Project Context numbers directly when tools are unavailable
- These changes eliminate generic UI-navigation answers for data questions — the assistant now responds with concrete numbers in all states (pre-simulation, post-simulation, pre-HM, post-HM)

**Desktop** — no binary change; shortcut remains `win-x64-v9`

---

### v2.0.3 (2026-03-14) — Dashboard, AI grounding & project persistence fixes

**Desktop (165/165 ViewModel tests pass)**

Dashboard fixes:
- **Wells card** — dashboard now shows the correct well count immediately after saving a new project via the wizard; loading a pre-v2.0.3 `.pfproj` (empty wells array) falls back to the Norne field default (31 wells) instead of showing 0
- **PINO trained status** — replaced the indirect `HasBeenTrained` / `TrainingStatusText` chain with an explicit `IsPinoTrained` observable flag; `MainWindowViewModel` sets it the moment `IsTraining` flips to `false` with a finite best loss — the "Trained ✓" card now appears reliably regardless of navigation path
- **Well serialisation** — `BuildProjectJson()` now writes the actual `Wells` collection to `"wells": [...]` in `.pfproj` files (was always `"wells": []`)
- **Project shortcut** — desktop shortcut updated to `win-x64-v9`

AI Assistant fixes:
- **System prompt rewritten** — explicit tool-routing table maps common question patterns to required tool calls; prohibited generic UI-navigation answers for data questions
- **Context enrichment** — `get_project_summary()` now inlines peak WOPR and water-cut per well for the above/below-expectation groups so models without tool-calling still quote real numbers
- **Project grounding** — `context_provider.set_project()` now parses `.pfproj` JSON and seeds full Norne baseline profiles (22 producers, 9 injectors, 37 timesteps) with `above_expectation` / `below_expectation` / `on_target` status labels
- **`get_well_performance("all")`** — returns compact per-well summary (peak WOPR, cumulative oil, water cut, status) instead of raw time-series
- **`get_data_mismatch_per_well()`** — adds `above_expectation`, `below_expectation`, `on_target` groupings plus `worst_wells` / `best_wells`
- **Project path forwarding** — `AgentServicer.Chat()` now correctly forwards `request.context_project` to `agent.chat(project_path=...)` (was silently dropped)
- **Helper consolidation** — `_water_cut` and `_cumulative` helpers unified into a single section in `ReservoirTools`

### v2.0.2 (2026-03-14) — Bug-fix release

**Test suite: 62/62 tests pass (was 33 failures + 9 errors)**

Core physics fixes:
- **`BlackOilPVT.Bo()`** — corrected oil FVF formula; was physically inverted giving Bo < 1 at reservoir pressure
- **`BlackOilPVT.Bg()`** — corrected gas FVF formula; was inverted (Bg increased with pressure instead of decreasing)
- **`gaspari_cohn()`** — fixed asymmetry; negative distances now handled via `jnp.abs()` so GC(−d) = GC(d)
- **`ReservoirGrid.transmissibility_x/y/z()`** — fixed to return face-centred arrays `(nx−1, ny, nz)` using harmonic-mean permeability instead of cell-centred `np.roll` arrays

Grid & well model:
- **`GridConfig.__post_init__`** — added validation rejecting non-positive nx/ny/nz
- **`ReservoirGrid`** — added `n_cells`, `n_active_cells` properties and `flatten()` / `unflatten()` convenience aliases
- **`Perforation`** — added `skin` and `wellbore_radius` fields
- **`WellConfig`** — added `bhp_limit` field and `is_injector()` helper
- **`WellType`** — added generic `INJECTOR` enum value (alongside `WATER_INJECTOR`, `GAS_INJECTOR`)
- **`PeacemannWellModel`** — updated per-well API: `productivity_index(well, k)`, `compute_oil_rates(well, pressure, bhp, …)`, `compute_injection_rates(well, pressure, bhp_inj, …)`; constructor now accepts optional `wells` list
- **`parse_compdat()`** — now accepts multi-line string or list of strings; returns `list[WellConfig]` with perforation objects

REST API & database:
- **`GET /api/v1/simulation/status`** — fixed `AttributeError` (`context.simulation_status` → `context.simulation_state`)
- **`AuditLogSchema.id`** — fixed type mismatch (`str` → `int`)
- **`POST /api/v1/simulation/run`** — implemented missing `_run_forward_surrogate` in `simulation_service.py`
- **`POST /api/v1/training/start`** — fixed FK constraint error; `pretrain_norne()` now accepts `project_id` and `run_id` parameters
- **`POST /api/v1/hm/start`** — implemented missing `_run_areki` in `hm_service.py`

History matching:
- **`AREKIEngine.__init__`** — added `observations` / `obs_error_cov` parameter aliases for test/gRPC compatibility
- **`AREKIEngine`** — added `_kalman_update_numpy`, `_svd_solve_numpy`, `_compute_alpha_numpy` pure-NumPy wrappers
- **`PVTConfig`** — added `api_gravity` and `gas_gravity` fields to `norne_defaults()`

---

## Table of Contents

1. [Changelog](#changelog)
2. [What Is PhysicsFlow?](#what-is-physicsflow)
3. [Key Capabilities](#key-capabilities)
4. [Architecture Overview](#architecture-overview)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Database Layer](#database-layer)
7. [Scientific Background](#scientific-background)
8. [Installation](#installation)
9. [Quick Start](#quick-start)
10. [AI Reservoir Assistant (Ollama)](#ai-reservoir-assistant)
11. [Project File Format (.pfproj)](#project-file-format)
12. [Running Unit Tests](#running-unit-tests)
13. [Industry Compliance](#industry-compliance)
14. [Competitive Positioning](#competitive-positioning)
15. [Roadmap](#roadmap)
16. [References](#references)

---

## What Is PhysicsFlow?

PhysicsFlow is an industrial reservoir simulation and history matching platform that replaces
classical finite-volume simulators (OPM FLOW, Eclipse 100) in the history matching loop with a
**Physics-Informed Neural Operator (PINO)** surrogate, achieving a **6,000× speed-up** while
maintaining physical consistency.

It implements and extends the methodology from:

> *"Reservoir History Matching of the Norne Field with Generative Exotic Priors and a Coupled
> Mixture of Experts — Physics Informed Neural Operator Forward Model"*
> Etienam et al., NVIDIA, arXiv:2406.00889v1 (2024)

Key innovations over the paper:

- **Hybrid PyTorch + JAX** engine: PyTorch for PINO training (FNO architecture), JAX/XLA for
  αREKI ensemble operations (3–5× faster Kalman updates via `jax.vmap` + `jax.jit`)
- **Local LLM assistant** (Ollama) embedded in the UI: ask questions about your reservoir in
  plain English, get data-grounded answers with live tool-calling
- **.NET 8 WPF desktop application**: professional UI comparable to REVEAL / Petex IPM Suite
- **Eclipse I/O**: native reader for .DATA / .EGRID / .UNRST / LAS 2.0 formats
- **Self-contained installer**: WiX v4 bootstrapper bundles Python, PyTorch wheels, and the
  .NET app into a single `PhysicsFlow-Installer-1.2.0-x64.exe`

---

## Key Capabilities

| Capability | Detail |
|---|---|
| Forward simulation | PINO surrogate: P + Sw + Sg fields over full 3D grid, ~7 sec vs 12 hr |
| History matching | αREKI — adaptive regularised ensemble Kalman inversion (JAX) |
| Uncertainty quantification | Ensemble P10/P50/P90 with VCAE + DDIM generative priors |
| Well model | CCR (Cluster-Classify-Regress) XGBoost mixture of experts |
| Speed-up | ~6,000× vs OPM FLOW on the Norne 46×112×22 benchmark |
| AI assistant | Local LLM (Ollama `deepseek-r1:1.5b` default) with 10 live reservoir tool calls + tool-call fallback for non-tool-capable models |
| Hybrid RAG | ChromaDB dense + BM25 sparse + RRF fusion + cross-encoder reranking |
| Knowledge graph | Reservoir KG (networkx): 22 layers, 22 wells, 53 faults, 5 segments, 20-pattern NL query |
| Input formats | Eclipse .DATA / .EGRID / .UNRST, OPM, LAS 2.0 well logs, .pfproj |
| Output formats | Excel (ClosedXML), PDF (QuestPDF), VTK for ResInsight/Paraview |
| GPU acceleration | CUDA via PyTorch (PINO training) + XLA via JAX (ensemble ops) |
| Deployment | .NET 8 WPF desktop, Windows 10/11, self-contained WiX installer |
| Project management | .pfproj JSON project files capturing full study configuration |
| Production forecast | P10/P50/P90 fan charts, EUR, recovery factor — OxyPlot |
| Project wizard | 5-step guided setup: Grid → Wells → PVT → Schedule → Save |
| 3D reservoir viewer | Interactive HelixToolkit voxel renderer: pressure/Sw/K fields, well markers, animation |
| 2D cross-section viewer | I/J/K-plane slices via WriteableBitmap — Jet/Viridis/Seismic/Greys colormaps |
| Project file encryption | AES-256-GCM (.pfproj.enc) — PBKDF2 600k iterations, CLI encrypt/decrypt commands |
| Unit tests | pytest suite: PVT, grid, wells, Kalman, localisation, material balance |
| Database & audit | SQLite shared DB: projects, runs, epochs, HM iterations, audit log |
| CI/CD | GitHub Actions: Python 3.11/3.12 matrix, .NET desktop build, ruff lint, bandit security |
| PINO pre-training | physicsflow-pretrain CLI: synthetic Norne ensemble, Buckley-Leverett proxy targets |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      PhysicsFlow Desktop (.NET 8 WPF)                     │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐  │
│  │Dashboard │  │ Project  │  │  Training  │  │ History  │  │Forecast │  │
│  │          │  │  Setup   │  │  Monitor   │  │ Matching │  │P10/P50/ │  │
│  │  Stats   │  │ 5-step   │  │  (PINO)    │  │ (αREKI)  │  │  P90   │  │
│  │  Wells   │  │  Wizard  │  │  Loss curve│  │ Fan chart│  │  EUR   │  │
│  └──────────┘  └──────────┘  └────────────┘  └──────────┘  └─────────┘  │
│                                                                            │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  3D Reservoir Viewer       │  │  2D Cross-Section Viewer           │  │
│  │  HelixToolkit voxels       │  │  I/J/K slices, WriteableBitmap     │  │
│  │  Jet colormap, animation   │  │  Jet/Viridis/Seismic/Greys maps    │  │
│  └────────────────────────────┘  └────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │           AI Reservoir Assistant (streaming chat, tool calls)        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ gRPC bidirectional streaming
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│  SimulationSvc   │  │  HistoryMatching  │  │  AgentService           │
│  TrainingSvc     │  │  Service (αREKI)  │  │  (Ollama LLM)           │
│                  │  │                  │  │                          │
│  PINO surrogate  │  │  JAX ensemble    │  │  Tool calling:           │
│  (FNO3d)         │  │  Kalman update   │  │  get_simulation_status  │
│  VCAE + DDIM     │  │  Gaspari-Cohn    │  │  get_well_performance   │
│  CCR well model  │  │  localisation    │  │  get_hm_summary         │
│  PVT (PyTorch)   │  │  VCAE z-space    │  │  get_ensemble_stats     │
│  Eclipse I/O     │  │  update          │  │  get_field_property     │
│  LAS 2.0 reader  │  │                  │  │  explain_parameter      │
└──────────────────┘  └──────────────────┘  │  search_project_        │
                                             │    knowledge (RAG)      │
                                             │  query_reservoir_graph  │
                                             └────────────┬────────────┘
                                                          │
                    ┌─────────────────────────────────────┴──────────────────┐
                    │           Intelligence Layer (v1.4.0)                   │
                    │                                                         │
                    │  ┌──────────────────────┐  ┌────────────────────────┐  │
                    │  │  Hybrid RAG          │  │  Reservoir KG          │  │
                    │  │  ChromaDB (dense)    │  │  networkx MultiDiGraph │  │
                    │  │  BM25 sparse index   │  │  22 layers, 22 wells   │  │
                    │  │  RRF fusion          │  │  53 faults, 5 segments │  │
                    │  │  Cross-encoder rerank│  │  20-pattern NL engine  │  │
                    │  │  BGE-small-en-v1.5   │  │  JSON persistence      │  │
                    │  │  Multi-query / HyDE  │  │  4-source builder      │  │
                    │  └──────────────────────┘  └────────────────────────┘  │
                    └─────────────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
┌──────────────────────────────────────────────┐
│          ReservoirContextProvider             │
│  Thread-safe shared state (threading.RLock)  │
│  Written by services → read by agent tools   │
└──────────────────────────────┬───────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────┐
│    SQLite Database — physicsflow.db           │
│    %APPDATA%\PhysicsFlow\physicsflow.db       │
│                                               │
│  Python (SQLAlchemy — owns schema + writes)   │
│  ├── DatabaseService singleton (thread-safe)  │
│  ├── projects, simulation_runs, training_epochs│
│  ├── hm_iterations, well_observations         │
│  ├── model_versions, audit_log (append-only)  │
│                                               │
│  .NET (EF Core — read-optimised UI queries)   │
│  ├── AppDbService (async, per-request ctx)    │
│  └── 7 entity classes mirroring Python schema │
└──────────────────────────────────────────────┘
```

### Communication Layer

- **gRPC with Protocol Buffers**: type-safe, bidirectional streaming for training progress,
  αREKI iteration updates (with P10/P50/P90 preview), and LLM streaming tokens
- **Python subprocess management**: .NET `EngineManager` starts/stops the Python gRPC server
  automatically, waits for `engine.ready` signal file, redirects stdout/stderr to Serilog
- **Ollama HTTP API**: Python agent communicates with local Ollama server; .NET chat UI
  streams tokens via gRPC (`AgentService.Chat`) back to the user

---

## Technology Stack

### Python Engine

| Component | Technology | Version |
|---|---|---|
| Neural network training | PyTorch | ≥ 2.4.0 |
| Ensemble Kalman operations | JAX + BlackJAX | ≥ 0.4.28 |
| FNO/PINO architecture | Custom (fno.py, SpectralConv3d) | v1.1.0 |
| Physics loss | Custom PDE residuals (Darcy, Peacemann) | v1.1.0 |
| CCR well model | XGBoost + scikit-learn K-Means | ≥ 2.1.0 |
| Generative priors | VCAE + DDIM (PyTorch, priors.py) | v1.1.0 |
| gRPC server | grpcio + grpcio-tools | ≥ 1.64.0 |
| LLM agent | Ollama Python SDK (tool-calling) | ≥ 0.2.0 |
| Eclipse I/O | Native parser (eclipse_reader.py) | v1.1.0 |
| LAS I/O | Native parser (las_reader.py) | v1.1.0 |
| Configuration | Pydantic-settings (PHYSICSFLOW_* env vars) | ≥ 2.3.0 |
| Logging | Loguru | ≥ 0.7.0 |
| Database ORM | SQLAlchemy (WAL mode, thread-safe sessions) | 2.0.x |
| Migrations | Alembic (schema evolution) | 1.16.x |
| File encryption | cryptography (AES-256-GCM, PBKDF2-HMAC-SHA256) | ≥ 42.0.0 |
| PINO pre-training | pretrain_norne.py — synthetic ensemble, CLI entry point | v1.2.0 |
| RAG — vector store | ChromaDB (persistent, cosine similarity) | ≥ 0.5.0 |
| RAG — embeddings | sentence-transformers (BAAI/bge-small-en-v1.5, 512-dim) | ≥ 3.0.0 |
| RAG — sparse index | rank-bm25 (BM25Okapi) | ≥ 0.2.2 |
| RAG — reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | ≥ 3.0.0 |
| RAG — document I/O | PyMuPDF (PDF), python-docx (Word), chardet (TXT/CSV/LAS) | latest |
| Knowledge graph | networkx (MultiDiGraph — typed nodes/edges) | ≥ 3.3 |

### .NET Desktop

| Component | Technology | Version |
|---|---|---|
| UI framework | WPF + .NET 8 | 8.0 |
| MVVM | CommunityToolkit.MVVM | 8.x |
| UI controls | MahApps.Metro (dark theme) | 2.4.10 |
| Charts | OxyPlot.Wpf (fan charts, loss curves) | latest |
| Animated charts | LiveCharts2 | latest |
| 3D visualization | HelixToolkit.Wpf (voxel renderer, well tubes) | latest |
| gRPC client | Grpc.Net.Client | latest |
| PDF reports | QuestPDF Community (HM summary + EUR reports) | latest |
| Excel export | ClosedXML (wells, ensemble stats, training history) | latest |
| Logging | Serilog | latest |
| Database ORM | EF Core + Microsoft.EntityFrameworkCore.Sqlite | 8.0.x |
| Installer | WiX Toolset v4 | 4.x |

---

## Project Structure

```
PhysicsFlow/
├── README.md                              ← This file (v1.3.0)
├── CHANGELOG.md                           ← Full version history
├── build_pitchdeck.py                     ← python-docx pitch deck generator
│
├── engine/                                ← Python backend
│   ├── pyproject.toml                     ← Dependencies: torch, jax, grpcio, ollama, xgboost
│   ├── pytest.ini                         ← pytest configuration + markers
│   │
│   ├── physicsflow/
│   │   ├── __init__.py                    ← Public API exports
│   │   ├── server.py                      ← gRPC server entry point (click CLI)
│   │   ├── config.py                      ← Pydantic config (PHYSICSFLOW_* env vars)
│   │   │
│   │   ├── proto/                         ← Protocol Buffer definitions
│   │   │   ├── simulation.proto           ← SimulationService + TrainingService
│   │   │   ├── history_matching.proto     ← HistoryMatchingService
│   │   │   └── agent.proto                ← AgentService (streaming chat tokens)
│   │   │
│   │   ├── core/                          ← Core physics engine
│   │   │   ├── pvt.py                     ← BlackOilPVT: μg, Rs, Bo, Bg, μo, μw, Bw (PyTorch)
│   │   │   ├── grid.py                    ← ReservoirGrid: transmissibility, active mask
│   │   │   └── wells.py                   ← PeacemannWellModel, parse_compdat(), norne_default_wells()
│   │   │
│   │   ├── surrogate/                     ← AI forward model
│   │   │   ├── __init__.py
│   │   │   ├── fno.py                     ← FNO3d + PINO: SpectralConv3d, PINOLoss, train_one_epoch()
│   │   │   └── ccr.py                     ← CCRWellSurrogate: K-Means→XGBoost classifier→regressors
│   │   │
│   │   ├── history_matching/              ← Inverse problem (JAX)
│   │   │   ├── areki_jax.py               ← AREKIEngine: JIT Kalman update, adaptive α
│   │   │   ├── localisation_jax.py        ← gaspari_cohn(), build_localisation_matrix()
│   │   │   └── priors.py                  ← VCAE encoder + DDIMPrior + ReservoirPriorModel
│   │   │
│   │   ├── agent/                         ← LLM Reservoir Assistant
│   │   │   ├── reservoir_agent.py         ← ReservoirAgent: Ollama tool-calling, streaming, KG injection
│   │   │   ├── tools.py                   ← 10 agent tools: 8 live + search_project_knowledge + query_reservoir_graph
│   │   │   └── context_provider.py        ← ReservoirContextProvider (thread-safe RLock)
│   │   │
│   │   ├── rag/                           ← Hybrid RAG knowledge assistant (v1.3.0)
│   │   │   ├── __init__.py                ← Public exports
│   │   │   ├── document_processor.py      ← DocumentChunk, source processors (PDF/Word/TXT/CSV/LAS/Eclipse)
│   │   │   ├── vector_store.py            ← ChromaDB + BGE-small embeddings, cosine search, upsert/delete
│   │   │   ├── bm25_index.py              ← BM25Okapi sparse index, thread-safe, persist/load
│   │   │   ├── retriever.py               ← HybridRetriever: RRF fusion, cross-encoder rerank, HyDE expansion
│   │   │   ├── pipeline.py                ← RAGPipeline singleton: index_file/dir, search, clear
│   │   │   └── ingestor.py                ← FileIngestor: watch directory, incremental re-index
│   │   │
│   │   ├── kg/                            ← Reservoir Knowledge Graph (v1.3.0)
│   │   │   ├── __init__.py                ← Public exports
│   │   │   ├── graph.py                   ← ReservoirGraph: networkx MultiDiGraph, NodeType/EdgeType enums
│   │   │   ├── builder.py                 ← KGBuilder: Norne base + pfproj + DB + context population
│   │   │   ├── query_engine.py            ← KGQueryEngine: 20 regex patterns → typed graph traversal
│   │   │   ├── serializer.py              ← Atomic JSON save/load (%APPDATA%/PhysicsFlow/kg/)
│   │   │   └── pipeline.py                ← KGPipeline singleton: load-or-build, query, rebuild, update
│   │   │
│   │   ├── io/                            ← Data I/O
│   │   │   ├── __init__.py
│   │   │   ├── eclipse_reader.py          ← .DATA keyword + .EGRID/.UNRST binary reader
│   │   │   ├── las_reader.py              ← LAS 2.0 parser: all sections, resample, batch read
│   │   │   ├── project.py                 ← PhysicsFlowProject: .pfproj JSON save/load + encryption
│   │   │   └── crypto.py                  ← AES-256-GCM: encrypt/decrypt/is_encrypted, PFEC format
│   │   │
│   │   ├── training/                      ← PINO pre-training scripts
│   │   │   └── pretrain_norne.py          ← PretrainConfig, _build_dataset(), pretrain_norne() CLI
│   │   │
│   │   ├── cli/                           ← Click CLI commands
│   │   │   ├── __init__.py
│   │   │   └── encrypt_cmd.py             ← physicsflow-encrypt / physicsflow-decrypt
│   │   │
│   │   ├── db/                            ← SQLite persistence layer (SQLAlchemy)
│   │   │   ├── __init__.py                ← Public exports
│   │   │   ├── models.py                  ← 7 ORM models: Project, SimulationRun, TrainingEpoch,
│   │   │   │                              │   HMIteration, WellObservation, ModelVersion, AuditLog
│   │   │   ├── database.py                ← Engine (WAL mode), get_session() ctx manager, init_db()
│   │   │   ├── repositories.py            ← ProjectRepo, RunRepo, HMRepo, WellObsRepo, ModelRepo, AuditRepo
│   │   │   └── db_service.py              ← DatabaseService singleton (thread-safe facade)
│   │   │
│   │   └── services/                      ← gRPC service implementations
│   │       ├── __init__.py
│   │       ├── simulation_service.py      ← SimulationServicer + TrainingServicer
│   │       ├── hm_service.py              ← HistoryMatchingServicer
│   │       └── agent_service.py           ← AgentServicer
│   │
│   └── tests/                             ← pytest unit tests
│       ├── __init__.py
│       ├── test_pvt.py                    ← PVT range, monotonicity, gradients, batching
│       ├── test_grid.py                   ← Transmissibility shape/sign, flatten roundtrip
│       ├── test_wells.py                  ← PI, producer/injector rates, COMPDAT parser
│       ├── test_kalman.py                 ← Kalman shape, mismatch reduction, SVD solve
│       ├── test_localisation.py           ← Gaspari-Cohn BCs, symmetry, matrix shape
│       └── test_material_balance.py       ← Havlena-Odeh F, Eo, volume balance
│
├── desktop/                               ← .NET 8 WPF application
│   ├── PhysicsFlow.sln                    ← Visual Studio solution (4 projects)
│   ├── src/
│   │   ├── PhysicsFlow.App/               ← Startup WPF project
│   │   │   ├── App.xaml / App.xaml.cs     ← DI host, Serilog, graceful shutdown
│   │   │   ├── MainWindow.xaml            ← 3-column layout: sidebar + content + AI panel
│   │   │   └── Views/
│   │   │       ├── Dashboard/             ← Stat cards, well map, quick actions
│   │   │       ├── ProjectSetup/          ← 5-step wizard (Grid→Wells→PVT→Schedule→Review)
│   │   │       ├── Training/              ← PINO training monitor, live loss curves
│   │   │       ├── HistoryMatching/       ← αREKI workspace, fan chart, per-well heatmap
│   │   │       ├── Forecast/              ← P10/P50/P90 fan charts, EUR, export buttons
│   │   │       ├── AIAssistant/           ← Streaming chat, quick actions, typing indicator
│   │   │       └── Visualisation/         ← 3D reservoir viewer + 2D cross-section viewer
│   │   │
│   │   ├── PhysicsFlow.Core/              ← Domain models + interfaces
│   │   ├── PhysicsFlow.ViewModels/        ← MVVM ViewModels
│   │   │   ├── MainWindowViewModel.cs     ← Navigation, engine lifecycle, AI panel toggle
│   │   │   ├── AIAssistantViewModel.cs    ← Streaming chat, quick actions, model mgmt
│   │   │   ├── ForecastViewModel.cs       ← Fan charts, EUR stats, export commands
│   │   │   └── ProjectSetupViewModel.cs   ← 5-step wizard, COMPDAT import, .pfproj save
│   │   │
│   │   └── PhysicsFlow.Infrastructure/    ← gRPC client, engine manager, reports, export, DB
│   │       ├── Engine/EngineManager.cs    ← Python process lifecycle, engine.ready signal
│   │       ├── Agent/OllamaAgentClient.cs ← gRPC streaming chat client
│   │       ├── Reports/                   ← QuestPDF report generation
│   │       │   ├── IReportService.cs      ← HM summary + EUR report interfaces
│   │       │   └── ReportService.cs       ← QuestPDF Community: tables, charts, disclaimer
│   │       ├── Export/                    ← ClosedXML Excel export
│   │       │   ├── IExcelExportService.cs ← Well data, ensemble stats, training history interfaces
│   │       │   └── ExcelExportService.cs  ← Multi-sheet workbooks with styled headers
│   │       └── Data/                      ← EF Core read layer (shared SQLite)
│   │           ├── PhysicsFlowDbContext.cs← EF Core DbContext: 7 DbSets, indexes, FK cascade
│   │           ├── AppDbService.cs        ← Async UI query service (projects, runs, HM, wells)
│   │           └── Entities/              ← 7 entity classes mirroring Python ORM schema
│   │               ├── ProjectEntity.cs
│   │               ├── SimulationRunEntity.cs
│   │               ├── TrainingEpochEntity.cs
│   │               ├── HMIterationEntity.cs
│   │               ├── WellObservationEntity.cs
│   │               ├── ModelVersionEntity.cs
│   │               └── AuditLogEntity.cs
│   │
│   └── tests/                             ← .NET unit tests (planned v1.2)
│
└── installer/                             ← WiX v4 installer
    ├── PhysicsFlow.wxs                    ← MSI: file layout, registry, .pfproj association
    ├── PhysicsFlow.Bundle.wxs             ← Bootstrapper: .NET 8 + VC++ + MSI
    └── build.ps1                          ← PowerShell build script (dotnet publish → wix)
```

---

## Database Layer

PhysicsFlow uses a single **SQLite database** (`physicsflow.db`) shared between the Python engine
and the .NET desktop application. Python owns the schema and all writes; .NET performs read-only
queries for UI display.

### Database File Location

```
Windows: %APPDATA%\PhysicsFlow\physicsflow.db
         (C:\Users\<user>\AppData\Roaming\PhysicsFlow\physicsflow.db)
```

### Schema — 7 Tables

| Table | Rows written by | Purpose |
|---|---|---|
| `projects` | Python `ProjectRepo` | Project registry — name, grid dims, HM status |
| `simulation_runs` | Python `RunRepo` | Every training / forward run with timing + loss |
| `training_epochs` | Python `RunRepo.add_epoch()` | Per-epoch losses for live loss curve |
| `hm_iterations` | Python `HMRepo` | Per-αREKI iteration mismatch, α, P10/P50/P90 snapshots |
| `well_observations` | Python `WellObsRepo` | Observed + simulated rates per well per timestep |
| `model_versions` | Python `ModelRepo` | Checkpoint registry with SHA-256, loss, is_active flag |
| `audit_log` | Python `AuditRepo` | Immutable append-only compliance log (UPDATE blocked) |

### Python Layer (SQLAlchemy)

```python
from physicsflow.db.db_service import DatabaseService

db = DatabaseService.instance()               # thread-safe singleton

# Register / update a project
db.register_project(project)

# Record a training run
run_id = db.start_run(project_id, "training", config)
db.record_epoch(run_id, epoch=10, loss_total=0.042, loss_pde=0.018, ...)
db.complete_run(run_id, loss_total=0.021)

# History matching
hm_run = db.new_hm_run_id(project_id)
db.record_hm_iteration(project_id, hm_run, iteration=5, mismatch=0.31, alpha=0.5)

# Audit
db.audit("project.opened", f"Opened {project.name}", project_id=project_id)
```

### .NET Layer (EF Core)

```csharp
// Injected via DI (singleton)
public class DashboardViewModel
{
    public DashboardViewModel(AppDbService db) { ... }

    async Task LoadAsync()
    {
        var projects = await _db.GetRecentProjectsAsync(limit: 20);
        var summary  = await _db.GetSummaryAsync();
        var epochs   = await _db.GetEpochHistoryAsync(runId);
        var wells    = await _db.GetWellNamesAsync(projectId);
    }
}
```

### Key Design Decisions

- **WAL journal mode**: allows simultaneous Python writes + .NET reads without locking
- **Foreign key cascade delete**: deleting a project removes all child rows automatically
- **Immutable audit log**: SQLAlchemy `before_update` event raises `RuntimeError` on any
  ORM-level UPDATE attempt — guarantees append-only compliance trail
- **Shared path resolution**: both Python (`_default_db_path()`) and .NET (`ResolveDbPath()`)
  resolve to the same `%APPDATA%\PhysicsFlow\physicsflow.db` path — zero configuration
- **Schema owned by Python**: EF Core opens with `Cache=Shared` and never calls `EnsureCreated`
  with migrations — Python `init_db()` / `create_all()` is the single source of truth

---

## Scientific Background

### Forward Problem — PINO Surrogate

Three-phase black-oil Darcy flow PDEs solved by the FNO surrogate:

```
∂(φ·So·Bo⁻¹)/∂t = ∇·(K·kro·Bo⁻¹/μo · ∇P) + qo
∂(φ·Sw·Bw⁻¹)/∂t = ∇·(K·krw·Bw⁻¹/μw · ∇P) + qw
∂(φ·Sg·Bg⁻¹ + φ·So·Rs·Bo⁻¹)/∂t = ∇·[K·(krg·Bg⁻¹/μg + Rs·kro·Bo⁻¹/μo)·∇P] + qg
```

The **FNO3d** surrogate maps static reservoir properties to dynamic state fields:

```
Input  [B, 6, Nx, Ny, Nz] : K_log, φ, P_init, Sw_init, x_coord, z_coord
Output [B, T, 2, Nx, Ny, Nz] : P(t), Sw(t)  for T timesteps
```

Trained with composite **PINO loss**:

```
L = w_data·L_data  +  w_pde·L_pde  +  w_ic·L_ic  +  w_bc·L_bc  +  w_well·L_well
```

where `L_pde` is the finite-difference Darcy residual computed on the predicted pressure field.

### Inverse Problem — αREKI

Adaptive Regularised Ensemble Kalman Inversion updates parameters (log-K, φ) to minimise
data mismatch against observed well production:

```
Kalman gain :  K  = Cyd · (Cdd + α·Γ)⁻¹        [SVD-based inversion]
Ensemble update :  m_new = m + K · (d_obs + η - G(m))
```

α is computed adaptively via the discrepancy principle and accumulated in `s_cumulative`.
The algorithm terminates when `s_cumulative ≥ 1` (Morozov convergence) or `max_iter` reached.

Gaspari-Cohn localisation suppresses spurious long-range correlations:

```
L[i,j] = GC(dist(param_i, obs_j) / radius)    ∈ [0, 1]
K_loc = L ⊙ K                                  [Schur product]
```

### Generative Priors (VCAE + DDIM)

The ensemble operates in latent space to preserve geological realism:

```
Encode :  K_log  →  VCAE encoder  →  z  ∈ ℝ^256    (μ, σ² via β-VAE)
αREKI  :  z_i    →  Kalman update  →  z_i'          (Gaussian in latent)
Decode :  z_i'   →  DDIM sampler   →  K_log'        (non-Gaussian, geologically plausible)
```

DDIM uses 50 deterministic inference steps (vs 1000 DDPM) for fast decoding with cosine schedule.

### CCR Well Surrogate

Three-stage Mixture of Experts replaces the Peacemann analytical model for non-Darcy,
multi-phase near-wellbore conditions:

```
1. Cluster  : K-Means (n=5) clusters well flowing conditions
2. Classify : XGBoost classifier assigns state → cluster label
3. Regress  : Per-cluster XGBoost regressors predict (WOPR, WWPR, WGPR)
```

Feature vector per well: BHP, reservoir pressure stats, saturations, K, φ, PI (13 features).

### PVT Correlations (PyTorch)

All PVT functions are differentiable PyTorch operations:

| Property | Correlation |
|---|---|
| Gas viscosity μg | Lee-Kesler |
| Solution GOR Rs | Standing |
| Oil FVF Bo | Standing |
| Gas FVF Bg | Real gas law |
| Oil viscosity μo | Beggs & Robinson |
| Water viscosity μw | Modified McCain |
| Water FVF Bw | Meehan |

---

## Installation

### Prerequisites

- Windows 10/11 (64-bit, build 19041+)
- NVIDIA GPU recommended (CUDA 12.x) — CPU fallback supported
- Ollama installed locally for AI assistant (`ollama pull deepseek-r1:1.5b`)

### Production Installer (end users)

```
PhysicsFlow-Installer-1.3.0-x64.exe
  ├── .NET 8 Desktop Runtime    (auto-installed if missing)
  ├── Visual C++ 2022 x64       (auto-installed if missing)
  ├── PhysicsFlow.msi
  │     ├── Desktop app (self-contained .NET 8)
  │     ├── Python 3.11 embedded + pre-packaged wheels
  │     └── .pfproj file association + Start Menu shortcut
  └── Post-install: pip install + grpcio proto stub generation
```

### Developer Setup

```bash
# 1. Clone repository
git clone https://github.com/Danny024/PhysicsFlow.git
cd PhysicsFlow

# 2. Python engine — create venv and install
cd engine
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # installs torch, jax, grpcio, ollama, xgboost, etc.

# 3. Generate gRPC stubs from proto definitions
cd physicsflow/proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. *.proto

# 4. Pull Ollama model for AI assistant (default — supports tool-calling)
ollama pull deepseek-r1:1.5b

# 5. Start Python gRPC engine
python -m physicsflow.server --port 50051 --log-level INFO

# 6. Build and run .NET desktop application
cd ../../desktop
dotnet restore
dotnet build
dotnet run --project src/PhysicsFlow.App

# 7. Run unit tests
cd ../engine
pytest tests/ -v
```

### Build Installer

```powershell
# Requires: WiX v4 (dotnet tool install --global wix), .NET 8 SDK
cd installer
.\build.ps1 -Configuration Release -Version 1.3.0
# Output: PhysicsFlow-Setup-1.3.0-x64.msi
#         PhysicsFlow-Installer-1.3.0-x64.exe
```

---

## Quick Start

### 1. Create a New Project

Launch PhysicsFlow → Click **New Project** on the dashboard.

The 5-step wizard guides you through:
1. **Grid** — enter Nx/Ny/Nz or import from an Eclipse `.DATA` file
2. **Wells** — import COMPDAT or click "Load Norne defaults" (22P + 9WI + 4GI)
3. **PVT** — set initial pressure, temperature, API gravity (or load Norne defaults)
4. **Schedule** — add production/injection control periods
5. **Review & Save** — saves as `ProjectName.pfproj`

### 2. Train the PINO Surrogate

Navigate to **Training** → configure epochs, learning rate, PDE weight → click **Start Training**.

The live chart streams training progress epoch by epoch. The trained model is saved to
`models/pino_latest.pt` and referenced in the project file.

### 3. Run History Matching

Navigate to **History Matching** → configure ensemble size (200), max iterations (20),
localisation radius (12 cells) → click **Run**.

Watch the live convergence chart and per-well mismatch heatmap update in real time.
The engine stops automatically when `s_cumulative ≥ 1` (converged).

### 4. View the Production Forecast

Navigate to **Forecast** → select well / quantity → click **Run Forecast**.

Four fan charts display P10/P50/P90 for oil rate, water rate, cumulative oil, and pressure.
EUR and recovery factor statistics are shown in the header cards.
Export to Excel or PDF using the header buttons.

### 5. Ask the AI Assistant

Click the AI button (bottom-right) to open the chat panel.
Type any question about your reservoir:

```
"What is the current data mismatch?"
"Which wells are underperforming in the P50 forecast?"
"Explain why the localisation radius was set to 12 cells."
"Show me the ensemble statistics for well B-2H."
```

---

## AI Reservoir Assistant

The built-in assistant (Ollama + local LLM) is grounded by **three complementary layers**:

1. **Live tool calls** — real-time simulation state (10 tools)
2. **Reservoir Knowledge Graph** — structured facts about Norne topology, wells, faults, parameters
3. **Hybrid RAG** — indexed project documents, well reports, technical papers

### Supported Models & Tool-Calling

| Model | Tool-calling | Size | Notes |
|---|---|---|---|
| `deepseek-r1:1.5b` | ✓ | 1.1 GB | **Default** — fast, reasoning, tool-capable |
| `deepseek-r1:14b` | ✓ | 9.0 GB | High quality reasoning, tool-capable |
| `llama3.1:8b` | ✓ | 4.7 GB | Strong all-rounder |
| `qwen2.5:7b` | ✓ | 4.4 GB | Good tool use |
| `phi3:mini` | ✗ | 2.2 GB | Chat only — auto-fallback, no live data |
| `mistral:7b` | ✓ | 4.1 GB | Fast, reliable |

Models that do not support Ollama tool-calling (e.g. `phi3:mini`) automatically fall back to
plain-chat mode — the assistant still responds, but cannot query live simulation data.

To pull the default model:

```bash
ollama pull deepseek-r1:1.5b
```

The model dropdown in the AI panel lists all **installed models first**, followed by
14 curated suggestions. Click the status dot/refresh button to re-scan after pulling a new model.

### Example Conversations

**User:** What is the current αREKI convergence status?
**AI:** History matching is at iteration 9 of 20. Mismatch has reduced from 1.84 → 0.21
(88% improvement). s_cumulative = 0.73. Currently not yet converged. The per-well
breakdown shows E-3H is the highest contributor to mismatch at RMSE 0.043.

**User:** Which wells are underperforming in the P50 forecast?
**AI:** Wells B-2H, D-1H, and E-3H show P50 production rates below the observed
reference by more than 20%. This may indicate local permeability overestimation
in the southeast fault block.

**User:** Which layers does well B-2H perforate?
**AI (KG):** Well B-2H perforates layers K9, K10, K11, K12 — located in segment B
(Central main segment, highest STOIIP).

**User:** Which injectors support producers in segment C?
**AI (KG):** Injectors C-4AH and C-4H support producers C-1H, C-2H, C-3H, and K-3H
in segment C via pressure maintenance.

**User:** What parameters most influence WWCT?
**AI (KG):** The parameters that most influence WWCT are: kr_oil, kr_water.

**User:** Explain the VCAE latent space encoding.
**AI (RAG):** The VCAE maps each permeability field (46×112×22 cells) to a compact 256-dimensional
Gaussian latent vector z. This means αREKI updates 256 parameters per ensemble member
instead of 113,344, and the Gaussian assumption required by Kalman methods is valid
in latent space. The DDIM decoder then maps z back to a geologically realistic K field...

### Agent Tools

| Tool | Layer | What It Does |
|---|---|---|
| `get_simulation_status()` | Live | Current run state, progress %, ETA |
| `get_well_performance(well_name)` | Live | WOPR / WWPR / WGPR time series + chart data |
| `get_hm_iteration_summary()` | Live | αREKI convergence metrics per iteration + chart |
| `get_ensemble_statistics(quantity, well)` | Live | P10/P50/P90 fan chart data |
| `get_data_mismatch_per_well()` | Live | Per-well RMSE breakdown |
| `get_field_property(prop, i, j, k)` | Live | Local K/φ/pressure/Sw at grid cell |
| `explain_parameter(name)` | Live | Built-in knowledge base (14 reservoir engineering parameters) |
| `get_project_summary()` | Live | Full project metadata, model paths, HM results |
| `query_reservoir_graph(question)` | KG | 20-pattern NL query over reservoir topology graph |
| `search_project_knowledge(query, top_k)` | RAG | Hybrid semantic search over indexed project documents |

### Indexing Documents for RAG

Drop any project document into the watch folder and it is auto-indexed:

```python
from physicsflow.rag import RAGPipeline

rag = RAGPipeline.instance()
rag.index_file("reports/Norne_field_study.pdf")      # PDF
rag.index_file("data/B-2H.las")                      # LAS well log
rag.index_directory("docs/")                         # whole folder
print(f"Indexed: {rag.count()} chunks")
```

Supported formats: PDF, Word (.docx), plain text, CSV, LAS 2.0, Eclipse .DATA keywords.

---

## Project File Format

PhysicsFlow uses `.pfproj` (JSON) as its native project format.

```json
{
  "version": "1.3.0",
  "name": "Norne Q4 2024",
  "created": "2024-11-01T12:00:00",
  "modified": "2024-11-15T09:30:00",
  "grid": {
    "nx": 46, "ny": 112, "nz": 22,
    "dx": 50.0, "dy": 50.0, "dz": 20.0,
    "depth": 2000.0
  },
  "pvt": {
    "initial_pressure_bar": 277.0,
    "temperature_c": 90.0,
    "api_gravity": 40.0,
    "gas_gravity": 0.7,
    "swi": 0.20
  },
  "wells": [ { "name": "B-1H", "well_type": "PRODUCER", "perforations": [...] } ],
  "schedule": [ { "well_name": "B-1H", "control_mode": "ORAT", "target_value": 5000 } ],
  "eclipse_deck_path": "C:/data/NORNE_ATW2013.DATA",
  "las_files": ["C:/data/B-1H.las"],
  "model_paths": {
    "pino": "models/pino_latest.pt",
    "ccr":  "models/ccr.pkl",
    "vcae": "models/vcae.pt",
    "ddim": "models/ddim.pt"
  },
  "hm_results": {
    "n_ensemble": 200, "n_iterations": 18,
    "best_mismatch": 0.21, "converged": true,
    "per_well_rmse": { "B-1H": 0.012, "B-2H": 0.031 }
  },
  "forecast": {},
  "notes": ""
}
```

**Python API:**
```python
from physicsflow.io.project import PhysicsFlowProject

# Create new
proj = PhysicsFlowProject.new("Norne Study")
proj.save("norne_study.pfproj")

# Load existing
proj = PhysicsFlowProject.load("norne_study.pfproj")
print(proj.summary())

# Create from Eclipse deck
proj = PhysicsFlowProject.from_eclipse("Norne Q4", "NORNE_ATW2013.DATA")
```

---

## Running Unit Tests

```bash
cd engine
pytest tests/ -v

# Run specific suites
pytest tests/test_pvt.py -v
pytest tests/test_kalman.py -v

# Skip JAX tests (if JAX not installed)
pytest tests/ -m "not jax" -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

### Test Coverage

| Test File | What It Validates |
|---|---|
| `test_pvt.py` | PVT range checks, monotonicity, PyTorch gradients, batch tensors |
| `test_grid.py` | Transmissibility shape/sign, flatten/unflatten roundtrip, Norne dims |
| `test_wells.py` | Peacemann PI scaling, producer/injector rates, Norne defaults, COMPDAT parser |
| `test_kalman.py` | Kalman output shape, mismatch reduction, SVD solve accuracy, adaptive α |
| `test_localisation.py` | GC boundary conditions (GC(0)=1, GC(2c)=0), symmetry, matrix shape/values |
| `test_material_balance.py` | Havlena-Odeh F positivity/zero, expansion Eo, cumulative closure, volume balance |

---

## Industry Compliance

| Requirement | Implementation |
|---|---|
| Audit trail | SQLite `audit_log` table: immutable append-only, user + hostname + timestamp + entity ref |
| Reproducibility | Seed stored in `.pfproj`; deterministic replay guaranteed |
| Data security | AES-256 project file encryption (v1.2 roadmap) |
| Units | Imperial (field) and Metric — configurable per project |
| Input standards | Eclipse .DATA / .EGRID / .UNRST; LAS 2.0 well logs |
| Output standards | Excel (ClosedXML), PDF (QuestPDF), VTK (ResInsight/Paraview) |
| Error logging | Serilog structured logs (.NET) + Loguru (Python); errors captured in `simulation_runs.error_message` |
| Physics validation | PDE residual quantified and available in project summary |
| Reserve certification | Positioned as fast-screening tool; full FVM (OPM FLOW) recommended for final booking |
| Installer | WiX v4 signed MSI — `Software\PhysicsFlow Technologies\PhysicsFlow` registry key |

---

## Competitive Positioning

| Feature | PhysicsFlow v1.3 | REVEAL (Intersect) | Petex IPM Suite | CMG IMEX |
|---|---|---|---|---|
| History matching speed | **~7 sec/run (PINO)** | Hours | Hours | Hours |
| History matching method | αREKI ensemble | Manual / ES-MDA | Manual | Manual |
| Generative priors | **VCAE + DDIM** | None | None | None |
| AI assistant | **Local LLM + RAG + KG** | None | None | None |
| License cost | **$15k/yr (SaaS)** | $150k–$400k/yr | $80k–$200k/yr | $120k–$350k/yr |
| Deployment | **Desktop + installer** | Cloud / HPC | Desktop | Desktop/HPC |
| Open source physics | **OPM FLOW reference** | Proprietary | Proprietary | Proprietary |
| Eclipse compatibility | **.DATA / .EGRID / .UNRST** | Full | Full | Full |
| LAS 2.0 support | **Yes** | Yes | Yes | Yes |

---

## Roadmap

### v1.0.0 — Foundation ✅ Complete

- [x] gRPC Protocol Buffer definitions (simulation, HM, agent)
- [x] Core physics: BlackOilPVT (PyTorch), ReservoirGrid, PeacemannWellModel
- [x] JAX αREKI engine with Gaspari-Cohn localisation
- [x] Ollama LLM agent with 8 live tool calls
- [x] .NET 8 WPF shell: MainWindow, Dashboard, Training, History Matching, AI Assistant
- [x] EngineManager (.NET) + gRPC client (OllamaAgentClient)
- [x] ReservoirContextProvider (thread-safe shared state)

### v1.1.0 — Full Stack ✅ Complete (this release)

- [x] FNO3d / PINO surrogate (fno.py) with composite PDE loss
- [x] CCR well surrogate (ccr.py) — K-Means + XGBoost
- [x] VCAE + DDIM prior models (priors.py) for non-Gaussian K fields
- [x] Eclipse .DATA / .EGRID / .UNRST reader (eclipse_reader.py)
- [x] LAS 2.0 well log reader (las_reader.py)
- [x] .pfproj project file format (project.py)
- [x] All 4 gRPC service handlers (simulation, training, HM, agent)
- [x] ForecastView: P10/P50/P90 fan charts, EUR, export
- [x] ProjectSetup wizard: 5-step Grid→Wells→PVT→Schedule→Review
- [x] ForecastViewModel + ProjectSetupViewModel
- [x] pytest unit tests: PVT, grid, wells, Kalman, localisation, material balance
- [x] WiX v4 MSI + bootstrapper bundle + PowerShell build script
- [x] Python database layer: SQLAlchemy ORM (7 tables), WAL mode, 6 repositories, DatabaseService singleton
- [x] .NET database layer: EF Core + SQLite, PhysicsFlowDbContext, 7 entity classes, AppDbService
- [x] Immutable audit log: SQLAlchemy `before_update` event prevents any ORM UPDATE on audit_log
- [x] Shared SQLite DB path (%APPDATA%\PhysicsFlow\physicsflow.db) — zero-config cross-process access
- [x] Private GitHub repository: https://github.com/Danny024/PhysicsFlow

### v1.2.0 — Visualisation & Reports ✅ Complete

- [x] Helix Toolkit 3D reservoir viewer (P, Sw, K animated) — `ReservoirView3D.xaml` + `ReservoirView3DViewModel.cs`
- [x] 2D cross-section viewer (I/J/K planes) — `CrossSectionView.xaml` + `CrossSectionViewModel.cs` (WriteableBitmap, 4 colourmaps)
- [x] QuestPDF report generation (HM summary, EUR report) — `ReportService.cs` (Community licence)
- [x] ClosedXML Excel export (well data, ensemble statistics, training history) — `ExcelExportService.cs`
- [x] Real gRPC stub generation in CI/CD pipeline — `.github/workflows/ci.yml` (python-engine + dotnet-desktop + lint + security jobs)
- [x] PINO pre-training on Norne reference dataset — `training/pretrain_norne.py` (`physicsflow-pretrain` CLI)
- [x] AES-256-GCM project file encryption — `io/crypto.py` + `cli/encrypt_cmd.py` (PBKDF2-HMAC-SHA256, 600k iterations)

### v1.3.0 — Intelligence Layer ✅ Complete (this release)

- [x] Hybrid RAG pipeline — ChromaDB dense vector search + BM25 sparse index + RRF fusion
- [x] BAAI/bge-small-en-v1.5 embeddings (512-dim, BGE instruction prefix for query-time quality)
- [x] Cross-encoder reranking — `cross-encoder/ms-marco-MiniLM-L-6-v2` for top-k precision
- [x] Multi-query and HyDE (Hypothetical Document Embedding) query expansion
- [x] Document processor — PDF (PyMuPDF), Word (.docx), TXT/CSV/LAS/Eclipse .DATA
- [x] RAG tool exposed to LLM — `search_project_knowledge(query, top_k)` with source attribution
- [x] Reservoir Knowledge Graph — `networkx.MultiDiGraph` with 9 NodeTypes and 8 EdgeTypes
- [x] Norne pre-population — 22 layers, 17 producers, 5 injectors, 53 faults, 5 segments, injector-producer support pairs, uncertain parameter graph
- [x] KG 4-source builder — base structural → pfproj enrichment → SQLite sync → live context RMSE
- [x] 20-pattern NL query engine — deterministic regex dispatch to typed graph traversal methods
- [x] `query_reservoir_graph` agent tool — zero-latency structured KG answers
- [x] KG auto-injection into system prompt — matching queries answered from graph before LLM reasoning
- [x] JSON persistence for KG — atomic save via `.tmp` → rename, `%APPDATA%\PhysicsFlow\kg\`
- [x] networkx dependency added to `pyproject.toml` (RAG group)
- [x] Bug fixes: SyntaxError in `layers_of_well()`, wrong `get_session()` call, `hashlib` import order, dead code removal

### v2.0 — On-Premise Scale-Out ✅ Complete (this release)

- [x] **FastAPI REST API** (`/api/v1`) — all engine capabilities exposed over HTTP for Jupyter,
      automation scripts, and third-party tools; runs as a daemon thread alongside gRPC
- [x] 10 REST route modules: health, projects, runs, simulation, training, history matching,
      models, I/O (upload/parse/export), agent chat (sync + SSE streaming), tNavigator bridge
- [x] **Dual database backend** — SQLite (single-user default) and PostgreSQL (team/server);
      URL-aware factory in `database.py`; SQLite WAL mode; PostgreSQL connection pooling
- [x] **API key authentication** — `X-API-Key` header; empty key = no auth (LAN single-user);
      non-empty = enforced for team deployments; `require_api_key` FastAPI dependency
- [x] **CORS middleware** — configurable origin list for Jupyter (8888), React (3000),
      Streamlit (8501)
- [x] **Docker on-premise deployment** — `nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04`
      multi-stage `Dockerfile`; `docker-compose.yml` (SQLite single-user);
      `docker-compose.postgres.yml` (PostgreSQL team stack); `docker/init_pg.sql`
- [x] **.env.example** — documented environment template for all configuration variables
- [x] **tNavigator bridge** (`io/tnavigator_bridge.py`) — keyword-based ASCII `.sim` parser;
      bidirectional conversion: `.sim` → PhysicsFlow summary + `.pfproj` JSON;
      REST endpoints: import, export, run (subprocess with 10-min timeout)
- [x] **PINO model registry** — `ModelVersion` ↔ `SimulationRun` FK fully linked;
      REST routes for list, get, activate, download checkpoint (FileResponse streaming)
- [x] **DatabaseService v2.0** — added `list_projects`, `get_project`, `update_project`,
      `delete_project`, `list_runs`, `get_run`, `get_epoch_history`, `list_models`,
      `get_model_by_id`, `activate_model`, `get_active_model`
- [x] **pyproject.toml** — bumped to v2.0.0; added `fastapi`, `uvicorn[standard]`,
      `python-multipart`, `psycopg2-binary`, `asyncpg`; new `physicsflow-rest` CLI entry point
- [x] SSE streaming chat (`POST /api/v1/agent/chat/stream`) — real-time token delivery for
      web UIs and Jupyter streaming cells
- [x] Background job model — simulation, training, and HM all return `run_id` immediately;
      execution in daemon threads; status queryable via live context + DB endpoints

### v2.1 — Full Cloud *(deferred — implement on customer demand)*

- [ ] Docker images published to container registry (GHCR / Docker Hub / ACR)
- [ ] Azure ML / AWS SageMaker GPU burst for large ensemble HM
- [ ] React web dashboard (project browser, live training charts, HM fan charts)
- [ ] Multi-tenant SaaS mode with user/org isolation and RBAC
- [ ] Object storage (Azure Blob / S3) for model checkpoints and uploaded data
- [ ] CI/CD pipeline: automated image build, push, and staging deployment on `main` push

---

## REST API Quick Reference (v2.0)

Base URL: `http://<engine-host>:8000/api/v1`  |  Interactive docs: `/docs`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Engine version, DB backend, ports |
| GET/POST | `/projects` | List / create projects |
| GET/PUT/DELETE | `/projects/{id}` | Get / update / delete project |
| GET | `/projects/{id}/audit` | Immutable audit log for a project |
| POST | `/simulation/run` | Start PINO forward simulation (async) |
| GET | `/simulation/status` | Live simulation state |
| POST | `/training/start` | Start PINO training job (async) |
| GET | `/training/status` | Live training state (epoch, loss) |
| POST | `/hm/start` | Start αREKI history matching (async) |
| GET | `/hm/{id}/iterations` | Per-iteration mismatch / P10-P90 |
| GET | `/models/projects/{id}` | List model checkpoints, newest first |
| GET | `/models/{id}/download` | Stream `.pt` checkpoint to client |
| POST | `/io/upload/{project_id}` | Upload Eclipse / LAS / pfproj file |
| POST | `/io/parse/eclipse/{id}` | Parse Eclipse deck → JSON metadata |
| POST | `/agent/chat` | Synchronous AI assistant (Jupyter) |
| POST | `/agent/chat/stream` | SSE streaming AI assistant |
| POST | `/tnav/import/{id}` | Parse tNavigator `.sim` deck |
| GET | `/tnav/export/{id}` | Export project to tNavigator `.sim` |

### Python client example

```python
import httpx

BASE = "http://localhost:8000/api/v1"
HEADERS = {}  # add {"X-API-Key": "secret"} for team mode

# Check engine health
r = httpx.get(f"{BASE}/health", headers=HEADERS)
print(r.json())  # {"status": "ok", "version": "2.0.0", ...}

# Start history matching
r = httpx.post(f"{BASE}/hm/start", json={
    "project_id": "norne-001",
    "n_ensemble": 200,
    "max_iterations": 20,
}, headers=HEADERS)
run_id = r.json()["run_id"]

# Poll iterations
r = httpx.get(f"{BASE}/hm/{run_id}/iterations",
              params={"project_id": "norne-001"}, headers=HEADERS)
for it in r.json():
    print(f"Iter {it['iteration']:3d}  mismatch={it['mismatch']:.4f}")
```

---

## References

1. Etienam et al. (2024). *Reservoir History Matching of the Norne Field with Generative
   Exotic Priors and a Coupled Mixture of Experts — PINO Forward Model*. arXiv:2406.00889.
2. Li et al. (2020). *Fourier Neural Operator for Parametric PDEs*. arXiv:2010.08895.
3. Iglesias (2016). *A regularising iterative ensemble Kalman method for PDE-constrained
   inverse problems*. Inverse Problems, 32(2), 025002.
4. Song et al. (2020). *Denoising Diffusion Implicit Models*. arXiv:2010.02502.
5. Kingma & Welling (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
6. Gaspari & Cohn (1999). *Construction of correlation functions in two and three dimensions*.
   Q. J. R. Meteorol. Soc., 125(554), 723–757.
7. Standing (1947). *A Pressure-Volume-Temperature Correlation for Mixtures of California Oils
   and Gases*. API Drilling and Production Practice.
8. OPM Project. *Open Porous Media FLOW*. https://opm-project.org
9. NVIDIA PhysicsNeMo (Modulus). https://github.com/NVIDIA/physicsnemo
10. WiX Toolset v4. https://wixtoolset.org
11. Norne Field Dataset. *SINTEF / OPM open benchmark*.
    https://github.com/OPM/opm-tests/tree/master/norne

---

*PhysicsFlow v2.0.0 — Built by the PhysicsFlow Technologies team.*
*Repository: [github.com/Danny024/PhysicsFlow](https://github.com/Danny024/PhysicsFlow) (private)*
*For issues and feature requests: [GitHub Issues](https://github.com/Danny024/PhysicsFlow/issues)*
