# PhysicsFlow Changelog

All notable changes to PhysicsFlow are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Released] — v2.0.0 — On-Premise Scale-Out

### Build: 2026-03-09

### Added — REST API (FastAPI)

- `physicsflow/api/app.py` — `create_rest_app(cfg, context, db_svc)` factory
  - Mounts all 10 router modules under `/api/v1`
  - `CORSMiddleware` with configurable origin list
  - Startup/shutdown lifespan hooks writing to audit log
  - `run_standalone()` entry point for headless REST-only deployments
- `physicsflow/api/auth.py` — `require_api_key` FastAPI dependency
  - No-op when `PHYSICSFLOW_REST_API_KEY` is empty (single-user LAN mode)
  - Enforces `X-API-Key` header in team mode; returns HTTP 403 on mismatch
- `physicsflow/api/schemas.py` — 18 Pydantic v2 request/response schemas
  - All ORM-compatible via `ConfigDict(from_attributes=True)`
  - `ProjectCreateRequest`, `ProjectSchema`, `ProjectListResponse`
  - `TrainingStartRequest`, `RunSchema`, `TrainingEpochSchema`
  - `HMStartRequest`, `HMIterationSchema` (correct ORM field names)
  - `ModelVersionSchema`, `AuditLogSchema`
  - `ChatRequest`, `ChatResponse`
  - `tNavigatorImportResponse`, `StatusResponse`, `JobSubmittedResponse`
- `physicsflow/api/routes/health.py` — `GET /health` (no auth)
- `physicsflow/api/routes/projects.py` — full CRUD with pagination
- `physicsflow/api/routes/runs.py` — list/get runs with type+status filters; epoch history
- `physicsflow/api/routes/simulation.py` — async run, live status, well data, base64 field arrays
- `physicsflow/api/routes/training.py` — async start/stop, live epoch/loss status
- `physicsflow/api/routes/history_matching.py` — async start/stop, iterations, ensemble P10/P50/P90
- `physicsflow/api/routes/models.py` — list, activate, download `.pt` checkpoint (FileResponse)
- `physicsflow/api/routes/io.py` — multipart upload, Eclipse deck parse, pfproj export
- `physicsflow/api/routes/agent.py` — synchronous chat + SSE streaming; lazy ReservoirAgent init
- `physicsflow/api/routes/tnavigator.py` — import `.sim`, export `.sim`, subprocess run

### Added — tNavigator Bridge

- `physicsflow/io/tnavigator_bridge.py` — bidirectional `.sim` ↔ `.pfproj` converter
  - Keyword-based ASCII parser: `DIMENS`, `WELSPECS`, `COMPDAT`, `WCONPROD`,
    `TSTEP`, `ACTNUM`, `PERMX/Y/Z`, `PORO`, `TOPS`, `DX/DY/DZ`
  - `TNavigatorBridge(sim_path)` — parse and expose `SimDeck`
  - `to_summary()` — returns grid dims, well names, n_timesteps, keywords found
  - `to_pfproj()` — converts deck to PhysicsFlow project dict
  - `from_pfproj(path)` — loads `.pfproj` and builds synthetic deck
  - `to_sim()` — renders deck back to Eclipse-compatible ASCII text
  - Temp file always cleaned up via `try/finally`

### Added — Dual Database Backend

- `physicsflow/db/database.py` — URL-aware factory replacing SQLite-only singleton
  - `_resolve_db_url()` checks `PHYSICSFLOW_DB_URL` env → `config.db_url` → `PHYSICSFLOW_DB_PATH` → OS default
  - SQLite: WAL mode + `check_same_thread=False` + `pool_pre_ping`
  - PostgreSQL: `pool_size=10`, `max_overflow=20`, `pool_recycle=1800`
  - `db_backend()` helper returning `"sqlite"` or `"postgresql"`
  - `reset_engine()` for test teardown

### Added — Docker On-Premise Deployment

- `engine/Dockerfile` — multi-stage `nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04` build
  - Stage 1: builder installs all Python wheels
  - Stage 2: runtime copies installed packages; health-check via `/api/v1/health`
- `engine/docker-compose.yml` — single-user stack (SQLite + Ollama)
  - GPU pass-through via NVIDIA Container Toolkit
  - Named volumes: `pf_data`, `ollama_data`
- `engine/docker-compose.postgres.yml` — team stack (PostgreSQL + Ollama)
  - `postgres:16-alpine` with health-check
  - `PHYSICSFLOW_REST_API_KEY` for team authentication
- `engine/docker/init_pg.sql` — PostgreSQL init: `uuid-ossp`, `pg_trgm` extensions
- `engine/.env.example` — documented environment template for all 15 config variables
- `engine/.dockerignore` — excludes venv, data volumes, generated stubs, IDE files

### Added — DatabaseService Methods

- `register_project_from_dict(data)` — creates project from request dict
- `list_projects(limit, offset)` — paginated project listing
- `update_project(project_id, **kwargs)` — returns updated `Project` ORM object
- `delete_project(project_id)` — cascade delete with audit log
- `list_runs(project_id, run_type, status, limit)` — filtered run listing
- `get_run(run_id)`, `get_epoch_history(run_id)`
- `list_models(project_id)`, `get_model_by_id(model_id)`
- `activate_model(model_id)` — deactivates siblings, writes audit log
- `get_active_model(project_id, model_type)`

### Added — Configuration

- `physicsflow/config.py` — three new setting groups in `EngineConfig`
  - Database: `db_url`, `team_mode`
  - REST API: `rest_enabled`, `rest_port`, `rest_host`, `rest_api_key`, `rest_cors_origins`
  - tNavigator: `tnavigator_exe`, `tnavigator_license_server`
  - Helpers: `is_postgres()`, `effective_team_mode()`

### Changed

- `physicsflow/server.py` — added `_start_rest_api()` launching Uvicorn as daemon thread
  alongside gRPC; gracefully skips if `fastapi`/`uvicorn` not installed
- `pyproject.toml` — version `2.0.0`; added `fastapi>=0.115.0`,
  `uvicorn[standard]>=0.30.0`, `python-multipart>=0.0.9`,
  `psycopg2-binary>=2.9.9`, `asyncpg>=0.29.0`; new `physicsflow-rest` CLI entry point
- `physicsflow/db/repositories.py` — `ProjectRepo.all_recent()` supports `offset`;
  `RunRepo.recent()` supports `run_type` and `status` filters;
  `RunRepo.start()` stores `n_ensemble`

### Fixed (Bug-Fix Pass — commit 025eaa9)

- `register_project_from_dict()` missing (`NameError` at runtime)
- `update_project()` returned `bool` instead of `Optional[Project]`
- `list_runs()` / `RunRepo.recent()` ignored `run_type` / `status` filter args
- `list_projects()` ignored `offset` pagination parameter
- `HMIterationSchema` used `eur_p10/p50/p90` (non-existent) — corrected to `p10_snapshot/p50_snapshot/p90_snapshot`
- `WellObservationSchema` field names were backwards (`wopr_obs` vs ORM `obs_wopr`)
- `TrainingEpochSchema` missing `loss_ic`, `loss_bc`, `learning_rate`, `gpu_util`
- `n_ensemble` was silently discarded instead of persisted to `SimulationRun`
- Temp file leaked in `TNavigatorBridge.from_pfproj()` on exception
- `complete_run()` f-string operator precedence (`f"x" f"y" if cond else ""`)

### Architecture Decisions — v2.0.0

- **On-premise over cloud**: reservoir data is commercially sensitive; keeping compute
  on the operator's own hardware eliminates data sovereignty risk and widens addressable market
- **Shared context object**: `ReservoirContextProvider` passed by reference to both gRPC
  servicers and the REST `app.state` — zero serialisation overhead, single source of truth
- **Daemon thread for REST**: REST server dies automatically when the gRPC process exits;
  no separate process management required for single-server deployments
- **v2.1 cloud deferred**: Azure ML burst, container registry, web dashboard, multi-tenant
  SaaS — explicitly deferred until customer demand materialises

---

## [Released] — v1.3.0 — Intelligence Layer

### Build: 2026-03-09

### Added — Hybrid RAG Pipeline

- `physicsflow/rag/vector_store.py` — ChromaDB dense vector store
  - `BAAI/bge-small-en-v1.5` embeddings (512-dim, BGE instruction prefix)
  - Persistent collection per project; upsert-safe
- `physicsflow/rag/sparse_store.py` — BM25 sparse index (`rank-bm25`)
  - Per-project index serialised to `%APPDATA%/PhysicsFlow/rag/`
  - Word-boundary tokenisation with lowercase normalisation
- `physicsflow/rag/retriever.py` — hybrid dense+sparse retriever
  - Reciprocal Rank Fusion (RRF, k=60) score fusion
  - Configurable `top_k_dense` / `top_k_sparse` / `top_k_fused`
- `physicsflow/rag/reranker.py` — cross-encoder reranker
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` joint scoring
  - Falls back gracefully if model unavailable
- `physicsflow/rag/query_processor.py` — query expansion
  - Multi-query generation (3 paraphrases via LLM)
  - HyDE (Hypothetical Document Embedding) expansion
- `physicsflow/rag/document_processor.py` — multi-format document ingestion
  - PDF (PyMuPDF), Word (.docx), TXT, CSV, LAS 2.0, Eclipse `.DATA`
  - Sliding-window chunking (512 tokens, 64 overlap)
- `physicsflow/rag/indexer.py` — `RAGIndexer` — orchestrates ingest pipeline
- `physicsflow/rag/context_builder.py` — formats retrieved chunks as LLM context
- `physicsflow/rag/pipeline.py` — `RAGPipeline` end-to-end query entry point

### Added — Reservoir Knowledge Graph

- `physicsflow/kg/graph.py` — `ReservoirKnowledgeGraph` (`networkx.MultiDiGraph`)
  - 9 `NodeType`s: Well, Layer, Fault, Segment, FluidContact, Zone, UncertainParameter, Observation, SimulationResult
  - 8 `EdgeType`s: PERFORATES, BOUNDED_BY, CONNECTED_TO, SUPPORTS, HAS_OBSERVATION, HAS_RESULT, CORRELATES_WITH, TRUNCATED_BY
  - `add_well()`, `add_layer()`, `add_fault()`, `add_connection()`, `add_observation()`
- `physicsflow/kg/builder.py` — 4-source KG construction pipeline
  - Layer 1: base structural (22 Norne layers, 53 faults, 5 segments)
  - Layer 2: pfproj enrichment (17 producers, 5 injectors, completions, connections)
  - Layer 3: SQLite sync (HM iterations, converged mismatch per well)
  - Layer 4: live RMSE injection from `ReservoirContextProvider`
- `physicsflow/kg/query_engine.py` — `KGQueryEngine` — 20-pattern NL query dispatch
  - Deterministic regex dispatch to typed graph traversal methods
  - Returns structured dict answers at near-zero latency
- `physicsflow/kg/serializer.py` — atomic JSON persistence (`.tmp` → rename)
- `physicsflow/kg/pipeline.py` — `KGPipeline` — build + query facade

### Added — Agent Intelligence Layer

- `physicsflow/agent/reservoir_agent.py` — upgraded to 3-layer grounding
  - Layer 1: KG auto-injection (matching queries answered from graph before LLM)
  - Layer 2: RAG context retrieval (hybrid dense+sparse+reranked)
  - Layer 3: 10 live tool calls (8 original + `search_project_knowledge` + `query_reservoir_graph`)
- `physicsflow/agent/tools.py` — 2 new tools
  - `search_project_knowledge(query, top_k)` — hybrid RAG search with source attribution
  - `query_reservoir_graph(question)` — structured KG traversal

### Fixed

- `SyntaxError` in `KGQueryEngine.layers_of_well()` (malformed walrus operator)
- Wrong `get_session()` call in `KGBuilder._sync_hm_data()`
- Redundant `import os, hashlib` inside `ModelRepo.register()` (shadowed module imports)
- `ReservoirAgent` constructed with wrong kwargs: `ollama_host=` (rejected) and `context=` (should be `context_provider=`)

---

## [Released] — v1.2.0

### Build: 2026-03-09

### Added — Python Engine

- `physicsflow/training/pretrain_norne.py` — PINO pre-training CLI and library
  - `PretrainConfig` dataclass with full hyperparameter control
  - `pretrain_norne(cfg)` function: generates synthetic Norne ensemble, trains FNO3d with PINOLoss
  - `_build_dataset()`: reads Eclipse deck if available, falls back to synthetic lognormal K/phi fields
  - `_synthetic_simulation()`: fast analytical Buckley-Leverett proxy for target field generation
  - `physicsflow-pretrain` CLI entry point (Click)
  - Integrates with `DatabaseService`: logs every epoch, registers best checkpoint in `model_versions`
- `physicsflow/io/crypto.py` — AES-256-GCM project file encryption
  - `encrypt_pfproj(path, password)` → `.pfproj.enc` (PBKDF2-HMAC-SHA256, 600k iterations, random salt)
  - `decrypt_pfproj(path, password)` → `.pfproj`
  - `is_encrypted(path)` → bool
  - PFEC binary format: magic + version + iterations + salt + nonce + tag + ciphertext
  - Secure deletion (`_secure_delete`) overwrites file with zeros before `unlink`
- `physicsflow/cli/encrypt_cmd.py` — CLI commands for encryption
  - `physicsflow-encrypt` — encrypt a project file (prompts password if not given)
  - `physicsflow-decrypt` — decrypt a `.pfproj.enc` file
- `physicsflow/io/project.py` — Updated for v1.2.0
  - `PhysicsFlowProject.save(path, password=None)` — optional AES-256-GCM encryption on save
  - `PhysicsFlowProject.load(path, password=None)` — transparent decryption of `.pfproj.enc` files
  - `PFPROJ_VERSION` bumped to `1.2.0`
- `pyproject.toml` — new dependencies: `sqlalchemy`, `alembic`, `cryptography`; three new CLI entry points

### Added — .NET Desktop

- `PhysicsFlow.Infrastructure/Reports/IReportService.cs` + `ReportService.cs`
  - QuestPDF-based PDF generation (Community licence)
  - `GenerateHMSummaryReportAsync`: project summary card, convergence table, per-well RMSE
  - `GenerateEURReportAsync`: EUR P10/P50/P90 table, per-well EUR, disclaimer block
  - Header/footer with page numbers and timestamp
- `PhysicsFlow.Infrastructure/Export/IExcelExportService.cs` + `ExcelExportService.cs`
  - ClosedXML-based Excel export
  - `ExportWellDataAsync`: Summary sheet + one sheet per well (obs/sim rates, all columns)
  - `ExportEnsembleStatisticsAsync`: EUR summary + per-well monthly P50 rates sheet
  - `ExportTrainingHistoryAsync`: epoch-by-epoch loss table with number formatting
- `Views/Visualisation/ReservoirView3D.xaml` + `ReservoirView3DViewModel.cs`
  - HelixToolkit.Wpf 3D reservoir viewer
  - Voxel-box rendering coloured by Jet colourmap for P / Sw / K
  - Animated timestep playback with Play/Pause commands and CancellationToken
  - Well trajectory overlay (5 representative Norne wells, yellow tubes)
  - Property selector, timestep slider, opacity slider, VTK export stub
- `Views/Visualisation/CrossSectionView.xaml` + `CrossSectionViewModel.cs`
  - I / J / K plane 2D bitmap slices using `WriteableBitmap`
  - Colourmaps: Jet, Viridis, Seismic, Greys
  - TabControl with three slice planes, well overlay Canvas
  - Slice index slider, property selector, colourmap selector
- `App.xaml.cs` — registers `IReportService`, `IExcelExportService`, `ReservoirView3DViewModel`, `CrossSectionViewModel` in DI
- `MainWindowViewModel.cs` — adds `NavigateTo3DViewer` and `NavigateToCrossSection` relay commands; injects both new ViewModels
- `MainWindow.xaml` — adds `Visualisation` namespace, `DataTemplate` entries for both new ViewModels; version updated to v1.2.0
- `ForecastViewModel.cs` — `ExportExcel` and `ExportPdf` commands wired to real services; `BuildEurReportData()` helper
- `PhysicsFlow.App.csproj` — `HelixToolkit.Wpf` already present from v1.1.0

### Added — DevOps / CI-CD

- `.github/workflows/ci.yml` — full GitHub Actions CI pipeline
  - `python-engine` job: Python 3.11 + 3.12 matrix, CPU torch, JAX CPU, generates gRPC stubs with `grpc_tools.protoc`, runs pytest (excludes gpu/jax/slow markers), uploads junit + coverage XML
  - `dotnet-desktop` job: Windows runner, `dotnet restore` triggers `Grpc.Tools` auto-codegen, `dotnet build` Release, `dotnet test`
  - `lint` job: ruff linter + formatter check
  - `security` job: bandit + pip-audit
- `.github/workflows/release.yml` — release automation on version tags (`v*.*.*`)
  - `build-installer` job: WiX v4 MSI + bootstrapper EXE, creates GitHub Release with assets
  - `publish-python` job: builds wheel, uploads to PyPI (requires `PYPI_TOKEN` secret)

### Changed

- `MainWindow.xaml` — version label updated from `v1.0.0-dev` to `v1.2.0`
- `pyproject.toml` — version bumped from `1.0.0-dev` to `1.2.0`

---

## [Unreleased] — v1.0.0-dev

### Build: 2025-03-09

### Added — Python Engine

#### Core Physics
- `physicsflow/core/pvt.py` — Black-oil PVT correlations as differentiable PyTorch operations
  - `BlackOilPVT`: μg, Rs, Bo, Bg, μo, μw, Bw
  - `PVTConfig` dataclass (Norne defaults included)
  - Unit conversion helpers (psia↔bar, STB/day↔m³/day)
- `physicsflow/core/grid.py` — Generalised reservoir grid object
  - `ReservoirGrid`: configurable Nx×Ny×Nz, transmissibility computation
  - Active cell masking, flatten/unflatten helpers
  - PyTorch tensor export via `.to_torch(device)`
  - Replaces hardcoded 46×112×22 Norne dimensions
- `physicsflow/core/wells.py` — Configuration-driven well model
  - `WellConfig`, `Perforation` dataclasses
  - `PeacemannWellModel`: productivity index J, rate computation
  - `parse_compdat()`: Eclipse COMPDAT keyword parser (reads any field's wells)
  - `norne_default_wells()`: Norne 22P+9WI+4GI configuration (0-based, from paper)
  - Replaces hardcoded (i,j) producer table in original code

#### Hybrid JAX/PyTorch History Matching Engine
- `physicsflow/history_matching/areki_jax.py` — JAX αREKI implementation
  - JIT-compiled Kalman update step (`jax.jit`)
  - SVD-based matrix inversion for numerical stability
  - Adaptive α via discrepancy principle
  - Convergence: stops when s_cumulative ≥ 1 or max_iter reached
  - 3–5× faster than original PyTorch loop via `jax.jit`
- `physicsflow/history_matching/localisation_jax.py` — Gaspari-Cohn localisation
  - `gaspari_cohn()`: 5th-order piecewise rational function
  - `build_localisation_matrix()`: N_params × N_obs localisation matrix
  - `well_observation_coords()`: maps well perforations to observation coords
  - `parameter_coords_3d()`: 3D grid → coordinate array for any grid size

#### AI Reservoir Assistant
- `physicsflow/agent/reservoir_agent.py` — Ollama LLM agent with tool-calling
  - Streaming token-by-token response via generator
  - Agentic loop: calls tools until no more tool calls needed
  - Session-based conversation history (last 20 messages retained)
  - System prompt: senior reservoir engineering expert persona
  - Mock response fallback when Ollama is not installed
- `physicsflow/agent/tools.py` — 8 agent tools
  - `get_simulation_status()` — current run state
  - `get_well_performance(well_name)` — WOPR/WWPR/WGPR with chart data
  - `get_hm_iteration_summary()` — αREKI convergence metrics + chart
  - `get_ensemble_statistics(quantity, well_name)` — P10/P50/P90 fan chart
  - `get_data_mismatch_per_well()` — per-well RMSE breakdown
  - `get_field_property(prop, i, j, k)` — local K/φ/pressure/saturation
  - `explain_parameter(name)` — built-in knowledge base (14 parameters)
  - `get_project_summary()` — full project metadata
  - All tools return JSON + optional `chart` data for UI rendering
- `physicsflow/agent/context_provider.py` — Thread-safe shared state store
  - Bridges live simulation results to agent tools
  - Written by gRPC services, read by agent tools
  - Thread-safe via `threading.RLock`

#### gRPC Interface
- `physicsflow/proto/simulation.proto` — Simulation + Training service definitions
  - `SimulationService`: RunForward, GetFieldSnapshot, GetWellTimeSeries, ValidateInputs
  - `TrainingService`: TrainSurrogate (streaming), LoadModel, SaveModel, EvaluateSurrogate
  - Full message types: GridSpec, WellSpec, PvtSpec, RelPermSpec, FieldSnapshot, WellResult
- `physicsflow/proto/history_matching.proto` — αREKI service definitions
  - `HistoryMatchingService`: RunHistoryMatch (streaming), GetEnsembleStats,
    GetDataMismatch, GetParameterEnsemble, StopHistoryMatch
  - Streaming `HMProgress` with per-iteration metrics + P10/P50/P90 preview
- `physicsflow/proto/agent.proto` — AI assistant service definitions
  - `AgentService`: Chat (streaming tokens), ListModels, SetModel, ClearHistory, GetToolLog
  - `ChatToken` message: token, tool call notifications, chart data, done signal
  - `ChartData` / `ChartSeries` for embedded visualisation

#### Infrastructure
- `physicsflow/server.py` — gRPC server entry point (click CLI)
  - Starts all 4 services on one port
  - Writes `engine.ready` signal file for .NET EngineManager
  - Serilog-compatible structured logging
  - Graceful SIGTERM shutdown
- `physicsflow/config.py` — Pydantic-based configuration
  - All settings from env vars (`PHYSICSFLOW_*`) or `.env` file
  - No hardcoded values; fully configurable per deployment
- `physicsflow/__init__.py` — clean public API
- `pyproject.toml` — full dependency specification (PyTorch, JAX, gRPC, Ollama, XGBoost)

### Added — .NET Desktop Application

#### Solution Structure
- `PhysicsFlow.sln` — Visual Studio solution with 4 projects
- `PhysicsFlow.App` (.csproj) — WPF startup project
  - MahApps.Metro (dark theme, Fluent controls)
  - OxyPlot.Wpf (production charts)
  - LiveCharts2 (animated charts)
  - Helix Toolkit (3D reservoir viewer)
  - Serilog (structured logging)
- `PhysicsFlow.Core` (.csproj) — Domain models
- `PhysicsFlow.ViewModels` (.csproj) — MVVM (CommunityToolkit.Mvvm)
- `PhysicsFlow.Infrastructure` (.csproj) — gRPC client, engine manager

#### Application Shell
- `App.xaml` + `App.xaml.cs` — DI host, Serilog configuration, graceful shutdown
- `MainWindow.xaml` — 3-column layout: sidebar + main content + AI panel
  - Collapsible AI assistant panel (floating toggle button)
  - Engine status indicator (green/amber/red pill)
  - Project name in sidebar

#### Views
- `Views/Dashboard/DashboardView.xaml` — Home screen
  - Stat cards: wells, grid, surrogate status, HM status
  - Quick action buttons: New/Open project, Train, History Match
  - Recent projects list + 2D well map (OxyPlot scatter)
- `Views/Training/TrainingView.xaml` — PINO training monitor
  - Configuration panel: mode, epochs, LR, PDE weight, GPU toggle
  - Live progress bar + stat row (total/PDE/data loss, GPU util, ETA)
  - Live loss curve chart (OxyPlot)
- `Views/HistoryMatching/HistoryMatchingView.xaml` — αREKI workspace
  - αREKI config: ensemble size, max iter, localisation radius, CCR, VCAE toggles
  - Live stat cards: iteration, mismatch, α, improvement %
  - Per-well mismatch heatmap (colour-coded chips)
  - Convergence chart + P10/P50/P90 fan chart with well/quantity selectors
- `Views/AIAssistant/AIAssistantView.xaml` — Chat panel
  - Model selector + Ollama status indicator
  - 7 quick action chips (Summarise HM, Well performance, Mismatch analysis, etc.)
  - Streaming message bubbles (user right / assistant left)
  - Animated typing indicator (bouncing dots)
  - Enter to send, Shift+Enter for newline

#### ViewModels
- `MainWindowViewModel.cs` — navigation, engine lifecycle, AI panel toggle
- `AIAssistantViewModel.cs` — streaming chat, quick actions, model management
  - `ChatMessage` / `ChatRole` / `QuickAction` domain types
  - Token-by-token streaming with `IAsyncEnumerable`
  - Tool call inline notifications
  - Conversation history management

#### Infrastructure
- `Engine/EngineManager.cs` — Python process lifecycle manager
  - Starts bundled (production) or dev Python engine
  - Waits for `engine.ready` signal file
  - Redirects engine stdout/stderr to .NET logger
  - Graceful SIGTERM + force-kill fallback
  - `EngineStatus` events for UI binding
- `Agent/OllamaAgentClient.cs` — gRPC streaming client for AI assistant
  - `ChatStreamAsync()`: `IAsyncEnumerable<ChatTokenResult>` streaming
  - `ListModelsAsync()`, `SetModelAsync()`, `ClearHistoryAsync()`
  - Chart data mapping from proto to C# records
  - Auto-reconnect on first use

### Added — Documentation
- `README.md` — Full project documentation (12 sections)
  - Architecture diagram (ASCII)
  - Feature comparison table vs REVEAL / Petroleum Experts IPM
  - Technology stack tables (Python + .NET)
  - Project structure tree
  - Scientific background (PDE, αREKI, VCAE/DDIM, CCR equations)
  - Installation guide (dev + production)
  - AI assistant example conversations + tool table
  - Industry compliance table
  - Full v1.0 → v2.0 roadmap
- `CHANGELOG.md` — This file

### Architecture Decisions
- **Hybrid PyTorch + JAX**: PyTorch for PINO training (leverages PhysicsNeMo),
  JAX for αREKI ensemble operations (3–5× faster via `jax.jit`)
- **gRPC streaming**: enables real-time UI updates for training/HM progress
  without polling; bidirectional streaming for LLM tokens
- **Ollama tool-calling**: agent calls live data tools before answering;
  responses grounded in actual simulation results, not hallucinations
- **Config-driven**: all previously hardcoded values (grid dims, well locations,
  PVT constants) now loaded from configuration or Eclipse decks
- **`engine.ready` signal file**: simple cross-process startup synchronisation
  without requiring complex IPC before gRPC is available

---

## [Released] — v1.1.0

### Build: 2025-03-09

### Added — Python Engine

#### Surrogate Models
- `physicsflow/surrogate/fno.py` — Fourier Neural Operator (FNO) + PINO architecture
  - `SpectralConv3d`: 3-D spectral convolution with 4-octant Fourier modes
  - `FNOLayer3d`: spectral conv + pointwise residual + InstanceNorm + GELU
  - `FNO3d`: full model — lifting → n_layers FNO → projection
  - `PINOLoss`: composite loss (data + PDE + IC + BC + well), weights configurable
  - `darcy_pde_residual()`: finite-difference Darcy flow residual for PINO
  - `build_input_tensor()`: assembles [K_log, φ, P_init, Sw_init, x_norm, z_norm]
  - `train_one_epoch()`: mixed-precision training loop with gradient clipping
  - `FNOConfig.norne()`: pre-tuned hyperparameters for 46×112×22 grid
- `physicsflow/surrogate/ccr.py` — CCR well surrogate (Cluster-Classify-Regress)
  - `WellState`: per-well feature vector (BHP, pressure, saturations, K, φ, PI)
  - `CCRWellSurrogate`: K-Means → XGBoost classifier → per-cluster XGBoost regressors
  - `predict()`, `predict_batch()` returning `WellRates` (q_oil, q_water, q_gas)
  - `build_training_dataset()`: converts OPM FLOW snapshots to training arrays
  - `save()` / `load()`: pickle serialisation

#### I/O Modules
- `physicsflow/io/eclipse_reader.py` — Eclipse simulation deck reader
  - `_read_binary_records()`: FORTRAN unformatted binary iterator (.EGRID, .UNRST)
  - `_read_keyword_records()`: ASCII .DATA keyword parser with repeat expansion
  - `EclipseReader`: reads DIMENS, PERMX/Y/Z, PORO, ACTNUM, WELSPECS, COMPDAT
  - `snapshots()`: reads all timesteps from .UNRST → `EclipseSnapshot` list
  - `to_training_arrays()`: exports directly to FNO training format
- `physicsflow/io/las_reader.py` — LAS 2.0 well log reader
  - `LASReader.read()`, `LASReader.read_string()` entry points
  - Parses ~VERSION, ~WELL, ~CURVE, ~PARAMETER, ~OTHER, ~ASCII sections
  - `WellLog`: curves, well_info, parameters, null replacement, wrap mode
  - `resample()`: interpolate all curves to new depth array
  - `read_las_directory()`: batch-reads all .las files in a folder
- `physicsflow/io/project.py` — .pfproj project file format (JSON)
  - `PhysicsFlowProject`: grid, pvt, wells, schedule, model_paths, hm_results, forecast
  - `save()` → UTF-8 JSON, `load()` → validated dataclass
  - `from_eclipse()`: creates project from Eclipse deck
  - `update_hm_results()`, `add_las_file()` convenience methods

#### History Matching Priors
- `physicsflow/history_matching/priors.py` — VCAE + DDIM prior models
  - `VCAEEncoder`: 4-layer downsampling CNN → (μ, log_σ²) latent
  - `VCAEDecoder`: transposed CNN decoder
  - `VCAE`: β-VAE loss, reparameterisation, encode/decode API
  - `DDIMScheduler`: cosine / linear noise schedule, DDIM sub-sequence
  - `DDIMUNet3d`: lightweight 3-D U-Net conditioned on timestep + latent z
  - `DDIMPrior`: training loss + `sample()` + `sample_ensemble()`
  - `ReservoirPriorModel`: encode/decode facade for αREKI integration

#### gRPC Service Handlers
- `physicsflow/services/simulation_service.py`
  - `SimulationServicer`: RunForward (PINO fast path + OPM FLOW fallback),
    GetFieldSnapshot, GetWellTimeSeries, ValidateInputs
  - `TrainingServicer`: TrainSurrogate (streaming epoch-by-epoch progress),
    LoadModel, SaveModel, EvaluateSurrogate
  - `_DummyDataset`: synthetic dataset for testing training pipeline
- `physicsflow/services/hm_service.py`
  - `HistoryMatchingServicer`: RunHistoryMatch (streaming αREKI with stop signal),
    GetEnsembleStats, GetDataMismatch, GetParameterEnsemble, StopHistoryMatch
- `physicsflow/services/agent_service.py`
  - `AgentServicer`: Chat (streaming tokens), ListModels, SetModel,
    ClearHistory, GetToolLog
- `physicsflow/server.py` — updated to wire all 4 services with `EngineConfig`

#### Unit Tests
- `tests/test_pvt.py` — PVT correlations: range checks, monotonicity, gradients, batching
- `tests/test_grid.py` — ReservoirGrid: transmissibility shape/sign, flatten roundtrip
- `tests/test_wells.py` — Peacemann PI, producer/injector rates, Norne defaults, COMPDAT parser
- `tests/test_kalman.py` — Kalman update shape, mismatch reduction, SVD solve, adaptive α
- `tests/test_localisation.py` — Gaspari-Cohn boundary conditions, symmetry, matrix shape
- `tests/test_material_balance.py` — Havlena-Odeh F, expansion Eo, volume balance
- `pytest.ini` — test configuration with markers (slow, gpu, jax)

### Added — .NET Desktop Application

#### Views
- `Views/Forecast/ForecastView.xaml` — P10/P50/P90 production forecast dashboard
  - EUR oil/gas stat cards with recovery factor and peak rate
  - Well/quantity/horizon selectors + historical data toggle
  - 2×2 chart grid: oil rate, water rate, cumulative oil, reservoir pressure
  - Fan chart shading + P10/P50/P90 legend
  - Export to Excel / PDF buttons
- `Views/ProjectSetup/ProjectSetupView.xaml` — 5-step project setup wizard
  - Step 1 Grid: manual Nx/Ny/Nz or Eclipse .DATA import
  - Step 2 Wells: COMPDAT import or manual entry + Norne defaults button
  - Step 3 PVT: initial pressure, temperature, API gravity, gas gravity, Swi
  - Step 4 Schedule: production/injection control periods DataGrid
  - Step 5 Review: summary text + project name + save path → .pfproj

#### ViewModels
- `ForecastViewModel.cs`
  - P10/P50/P90 fan chart data with OxyPlot models
  - EUR oil/gas, recovery factor, peak rate summary stats
  - `RunForecastCommand`, `ExportExcelCommand`, `ExportPdfCommand`
  - Well/quantity/horizon selection driving chart refresh
- `ProjectSetupViewModel.cs`
  - 5-step wizard navigation with validation per step
  - `BrowseEclipseDeckCommand`, `ImportCompdatCommand`, `LoadNorneDefaultsCommand`
  - `WellRow` / `ScheduleRow` ObservableObject grid row models
  - JSON .pfproj file generation + `ProjectSaved` event

#### Installer
- `installer/PhysicsFlow.wxs` — WiX v4 MSI installer
  - ProgramFiles64 layout: app/, engine/, models/, logs/
  - .pfproj file extension registration + shell open command
  - CustomAction: pip install Python deps + grpcio-tools proto stub generation
  - Major upgrade support (removes previous versions)
- `installer/PhysicsFlow.Bundle.wxs` — Bootstrapper bundle (.exe)
  - Chains: .NET 8 Desktop Runtime → VC++ 2022 x64 → PhysicsFlow.msi
- `installer/build.ps1` — PowerShell build script
  - dotnet publish (self-contained x64) → wix build MSI → wix build EXE

### Architecture Decisions — v1.1.0
- **Spectral convolution modes**: 12×12×6 Fourier modes for Norne grid — balances
  expressiveness vs. parameter count (64 channels × 4 layers = ~3M params)
- **CCR cluster count**: 5 clusters captures major flow regimes (high/low K,
  near/far injector, boundary effects) without overfitting
- **VCAE latent dim=256**: empirically sufficient for 46×112×22 K fields;
  reduces αREKI parameter space from 113,344 to 256 per cell pair
- **DDIM inference steps=50**: 20× faster than DDPM (1000 steps) with <1% quality loss
- **WiX v4**: replaces WiX v3 — modern CLI, .NET tooling, no IDE required
- **.pfproj as JSON**: human-readable, version-control friendly, forward-compatible

---

## Upcoming — v2.1 — Full Cloud *(deferred — implement on customer demand)*

### Planned
- [ ] Docker images published to container registry (GHCR / Docker Hub / ACR)
- [ ] Azure ML / AWS SageMaker GPU burst for large ensemble history matching
- [ ] React web dashboard (project browser, live training charts, HM fan charts)
- [ ] Multi-tenant SaaS mode with user/org isolation and RBAC
- [ ] Object storage (Azure Blob / S3) for model checkpoints and uploaded data
- [ ] CI/CD pipeline: automated image build, push, and staging deployment on `main` push

---

*Maintained by the PhysicsFlow development team.*
*Repository: [github.com/Danny024/PhysicsFlow](https://github.com/Danny024/PhysicsFlow)*
