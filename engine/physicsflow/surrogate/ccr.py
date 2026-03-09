"""
PhysicsFlow — CCR (Cluster-Classify-Regress) Well Surrogate Model.

Architecture:
    1. Cluster  — K-Means clusters well flowing conditions into regimes
    2. Classify — XGBoost classifier assigns each state to a cluster
    3. Regress  — Per-cluster XGBoost regressors predict well rates

This replaces the Peacemann analytical model for multi-phase, non-Darcy,
near-wellbore conditions where the analytical formula breaks down.

Reference:
    Norne benchmark, CCR methodology from SPE-182665
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WellState:
    """
    Input feature vector for the CCR surrogate.

    All values are at the perforated cells of one well.
    """
    pressure_bhp: float          # Pa — bottomhole pressure
    pressure_res: np.ndarray     # Pa — reservoir pressure at perfs [n_perfs]
    sw: np.ndarray               # water saturation at perfs [n_perfs]
    so: np.ndarray               # oil saturation at perfs [n_perfs]
    perm: np.ndarray             # mD — permeability at perfs [n_perfs]
    phi: np.ndarray              # porosity at perfs [n_perfs]
    pi: np.ndarray               # productivity index at perfs [n_perfs]

    def to_feature_vector(self) -> np.ndarray:
        """Flatten to 1-D feature array for ML models."""
        # Aggregate perfs to single well-level features
        return np.array([
            self.pressure_bhp,
            float(np.mean(self.pressure_res)),
            float(np.min(self.pressure_res)),
            float(np.max(self.pressure_res)),
            float(np.mean(self.sw)),
            float(np.mean(self.so)),
            float(np.mean(self.perm)),
            float(np.log10(np.mean(self.perm) + 1e-6)),
            float(np.mean(self.phi)),
            float(np.mean(self.pi)),
            float(np.sum(self.pi)),
            float(len(self.perm)),                           # n_active_perfs
            float(np.mean(self.pressure_res) - self.pressure_bhp),   # drawdown
        ], dtype=np.float32)


@dataclass
class WellRates:
    """Predicted well rates from CCR."""
    q_oil: float    # STB/day
    q_water: float  # STB/day
    q_gas: float    # Mscf/day


@dataclass
class CCRConfig:
    """Hyperparameters for the CCR model."""
    n_clusters: int = 5
    xgb_classifier_params: dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42,
    })
    xgb_regressor_params: dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
    })


# ─────────────────────────────────────────────────────────────────────────────
# CCR Model
# ─────────────────────────────────────────────────────────────────────────────

class CCRWellSurrogate:
    """
    Cluster-Classify-Regress well surrogate.

    Usage:
        ccr = CCRWellSurrogate(CCRConfig())
        ccr.fit(X_train, y_oil_train, y_water_train, y_gas_train)
        rates = ccr.predict(well_state)
    """

    def __init__(self, cfg: Optional[CCRConfig] = None):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "CCR requires scikit-learn and xgboost: "
                "pip install scikit-learn xgboost"
            )
        self.cfg = cfg or CCRConfig()
        self._fitted = False

        # Components
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=self.cfg.n_clusters,
                                n_init=10, random_state=42)
        self.classifier = xgb.XGBClassifier(**self.cfg.xgb_classifier_params)

        # One regressor per cluster × output (oil, water, gas)
        self.regressors: Dict[Tuple[int, str], xgb.XGBRegressor] = {}

    def fit(
        self,
        X: np.ndarray,          # [N, n_features]  — WellState.to_feature_vector()
        y_oil: np.ndarray,      # [N]
        y_water: np.ndarray,    # [N]
        y_gas: np.ndarray,      # [N]
    ) -> "CCRWellSurrogate":
        """
        Full CCR training pipeline.
        """
        # 1. Scale features
        X_scaled = self.scaler.fit_transform(X)

        # 2. Cluster
        cluster_labels = self.clusterer.fit_predict(X_scaled)

        # 3. Train classifier
        self.classifier.fit(X_scaled, cluster_labels)

        # 4. Train per-cluster regressors
        for c in range(self.cfg.n_clusters):
            mask = (cluster_labels == c)
            if mask.sum() < 5:
                # Too few samples — fall back to mean predictor
                for name, y in [('oil', y_oil), ('water', y_water), ('gas', y_gas)]:
                    mean_val = float(y[mask].mean()) if mask.sum() > 0 else 0.0
                    self.regressors[(c, name)] = _MeanPredictor(mean_val)
                continue

            for name, y in [('oil', y_oil), ('water', y_water), ('gas', y_gas)]:
                reg = xgb.XGBRegressor(**self.cfg.xgb_regressor_params)
                reg.fit(X_scaled[mask], y[mask])
                self.regressors[(c, name)] = reg

        self._fitted = True
        return self

    def predict(self, state: WellState) -> WellRates:
        """Predict well rates for a single WellState."""
        self._check_fitted()
        x = state.to_feature_vector().reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        cluster = int(self.classifier.predict(x_scaled)[0])
        return WellRates(
            q_oil=max(0.0, float(self.regressors[(cluster, 'oil')].predict(x_scaled)[0])),
            q_water=max(0.0, float(self.regressors[(cluster, 'water')].predict(x_scaled)[0])),
            q_gas=max(0.0, float(self.regressors[(cluster, 'gas')].predict(x_scaled)[0])),
        )

    def predict_batch(self, states: List[WellState]) -> List[WellRates]:
        """Predict for a list of WellState objects."""
        self._check_fitted()
        X = np.stack([s.to_feature_vector() for s in states])
        X_scaled = self.scaler.transform(X)
        clusters = self.classifier.predict(X_scaled)

        results = []
        for i, (state, cluster) in enumerate(zip(states, clusters)):
            x = X_scaled[i:i+1]
            results.append(WellRates(
                q_oil=max(0.0, float(self.regressors[(int(cluster), 'oil')].predict(x)[0])),
                q_water=max(0.0, float(self.regressors[(int(cluster), 'water')].predict(x)[0])),
                q_gas=max(0.0, float(self.regressors[(int(cluster), 'gas')].predict(x)[0])),
            ))
        return results

    def feature_importance(self) -> Dict[str, np.ndarray]:
        """Return per-cluster, per-output feature importances."""
        result = {}
        for (c, name), reg in self.regressors.items():
            if hasattr(reg, 'feature_importances_'):
                result[f'cluster{c}_{name}'] = reg.feature_importances_
        return result

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'ccr_model.pkl', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CCRWellSurrogate":
        """Load serialized model."""
        with open(Path(path) / 'ccr_model.pkl', 'rb') as f:
            return pickle.load(f)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CCRWellSurrogate must be fitted before predicting. "
                               "Call .fit() with training data first.")


# ─────────────────────────────────────────────────────────────────────────────
# Mean predictor (fallback for sparse clusters)
# ─────────────────────────────────────────────────────────────────────────────

class _MeanPredictor:
    """Trivial predictor returning a fixed mean — used for tiny clusters."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float32)

    @property
    def feature_importances_(self):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Training data builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    simulation_snapshots: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert OPM FLOW simulation snapshots to CCR training arrays.

    Each snapshot dict must contain:
        'wells'  — List[dict] with keys:
                   'bhp', 'pressure_res', 'sw', 'so', 'perm', 'phi', 'pi',
                   'q_oil', 'q_water', 'q_gas'

    Returns:
        X        : [N, n_features]
        y_oil    : [N]
        y_water  : [N]
        y_gas    : [N]
    """
    Xs, y_oils, y_waters, y_gases = [], [], [], []

    for snap in simulation_snapshots:
        for well in snap.get('wells', []):
            state = WellState(
                pressure_bhp=well['bhp'],
                pressure_res=np.asarray(well['pressure_res'], dtype=np.float32),
                sw=np.asarray(well['sw'], dtype=np.float32),
                so=np.asarray(well['so'], dtype=np.float32),
                perm=np.asarray(well['perm'], dtype=np.float32),
                phi=np.asarray(well['phi'], dtype=np.float32),
                pi=np.asarray(well['pi'], dtype=np.float32),
            )
            Xs.append(state.to_feature_vector())
            y_oils.append(float(well['q_oil']))
            y_waters.append(float(well['q_water']))
            y_gases.append(float(well['q_gas']))

    X       = np.stack(Xs) if Xs else np.empty((0, 13), dtype=np.float32)
    y_oil   = np.array(y_oils, dtype=np.float32)
    y_water = np.array(y_waters, dtype=np.float32)
    y_gas   = np.array(y_gases, dtype=np.float32)
    return X, y_oil, y_water, y_gas
