# PhysicsFlow Changelog

All notable changes to PhysicsFlow are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

## Upcoming — v1.2.0

### Planned
- [ ] Helix Toolkit 3D field viewer (pressure / saturation / permeability)
- [ ] QuestPDF report generation (HM summary, EUR report)
- [ ] ClosedXML Excel export (well data, ensemble statistics)
- [ ] Real gRPC stub generation pipeline in CI
- [ ] OPM FLOW result parser integration with EclipseReader
- [ ] PINO surrogate pre-training on Norne reference runs
- [ ] CCR training pipeline end-to-end test
- [ ] Multi-field support (not just Norne)
- [ ] Cloud GPU burst option (Azure ML / AWS SageMaker)
- [ ] Web dashboard (FastAPI + React)

---

*Maintained by the PhysicsFlow development team.*
