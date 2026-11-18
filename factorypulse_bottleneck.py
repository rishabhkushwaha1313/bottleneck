#!/usr/bin/env python3
"""
FactoryPulse AI - Production Bottleneck Prediction (single-file edition)
========================================================================

Everything needed for bottleneck prediction in one file, consolidated from the
project's ml/, src/simulator/, src/prediction/ and src/common/ modules:

    1. Config        - settings (environment variables override the defaults)
    2. Simulator     - 3-station factory (Cutting -> Drilling -> Packaging)
    3. Dataset       - synthetic runs + 60-second look-ahead bottleneck label
    4. Training      - Random Forest vs. rule-based baseline, episode-wise split
    5. Evaluation    - fresh unseen simulation runs
    6. Predictor     - real-time rolling-feature inference (optional MQTT)

Usage
-----
    python factorypulse_bottleneck.py generate            # build training CSV
    python factorypulse_bottleneck.py train               # train + save model
    python factorypulse_bottleneck.py evaluate            # test on fresh runs
    python factorypulse_bottleneck.py demo                # live-style prediction demo
    python factorypulse_bottleneck.py all                 # generate+train+evaluate+demo
    python factorypulse_bottleneck.py live                # MQTT mode (needs paho-mqtt)

Add --seed 42 to any command for reproducible results.

Requirements: numpy, pandas, scikit-learn, joblib   (paho-mqtt only for `live`)

Changes versus the original multi-file project (bug fixes):
  * Scenario clock is now *simulated* time. The original used wall-clock time, so
    in fast dataset generation scenario_b's gradual slowdown never developed.
  * Training and live inference now share ONE feature definition. Originally the
    predictor used different rolling/trend formulas (train/serve skew).
  * Added the missing `Path` import in the predictor.
  * Database and mandatory MQTT dependencies removed; MQTT is optional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FactoryPulse")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Optional: load a .env file if python-dotenv is installed (not required).
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class AppConfig:
    # Simulation
    SIMULATION_TIME_SCALE: float = float(_env("SIMULATION_TIME_SCALE", "1.0"))
    DEMAND_RATE_UNITS_PER_MIN: float = float(_env("DEMAND_RATE_UNITS_PER_MIN", "15.0"))
    MAX_QUEUE_CAPACITY: int = int(_env("MAX_QUEUE_CAPACITY", "50"))

    STATION1_BASE_RPM: float = float(_env("STATION1_BASE_RPM", "1800.0"))
    STATION1_CYCLE_TIME_SEC: float = float(_env("STATION1_CYCLE_TIME_SEC", "3.5"))
    STATION1_CAPACITY_PER_MIN: float = float(_env("STATION1_CAPACITY_PER_MIN", "17.0"))

    STATION2_BASE_RPM: float = float(_env("STATION2_BASE_RPM", "1500.0"))
    STATION2_CYCLE_TIME_SEC: float = float(_env("STATION2_CYCLE_TIME_SEC", "4.0"))
    STATION2_CAPACITY_PER_MIN: float = float(_env("STATION2_CAPACITY_PER_MIN", "15.0"))

    STATION3_BASE_RPM: float = float(_env("STATION3_BASE_RPM", "1200.0"))
    STATION3_CYCLE_TIME_SEC: float = float(_env("STATION3_CYCLE_TIME_SEC", "3.0"))
    STATION3_CAPACITY_PER_MIN: float = float(_env("STATION3_CAPACITY_PER_MIN", "20.0"))

    # Alert thresholds (also used to label bottlenecks)
    ALERT_RPM_CRITICAL_RATIO: float = float(_env("ALERT_RPM_CRITICAL_RATIO", "0.40"))
    ALERT_RPM_WARNING_RATIO: float = float(_env("ALERT_RPM_WARNING_RATIO", "0.70"))
    ALERT_QUEUE_WARNING_THRESHOLD: int = int(_env("ALERT_QUEUE_WARNING_THRESHOLD", "15"))
    ALERT_QUEUE_CRITICAL_THRESHOLD: int = int(_env("ALERT_QUEUE_CRITICAL_THRESHOLD", "30"))

    # Machine learning
    ML_MODEL_PATH: Path = PROJECT_ROOT / _env("ML_MODEL_PATH", "ml/models/bottleneck_rf_model.joblib")
    ML_METADATA_PATH: Path = PROJECT_ROOT / _env("ML_METADATA_PATH", "ml/models/model_metadata.json")
    DATASET_PATH: Path = PROJECT_ROOT / _env("DATASET_PATH", "data/synthetic/training_dataset.csv")
    ML_PREDICTION_HORIZON_SEC: int = int(_env("ML_PREDICTION_HORIZON_SEC", "60"))
    ML_ROLLING_WINDOW_SIZE: int = int(_env("ML_ROLLING_WINDOW_SIZE", "10"))
    ML_TREND_LAG: int = 5  # samples used for trend features

    # MQTT (only used by the `live` command)
    MQTT_MODE: str = _env("MQTT_MODE", "local").lower()  # "local" or "hivemq"
    MQTT_LOCAL_HOST: str = _env("MQTT_LOCAL_HOST", "127.0.0.1")
    MQTT_LOCAL_PORT: int = int(_env("MQTT_LOCAL_PORT", "1883"))
    MQTT_HIVEMQ_HOST: str = _env("MQTT_HIVEMQ_HOST", "")
    MQTT_HIVEMQ_PORT: int = int(_env("MQTT_HIVEMQ_PORT", "8883"))
    MQTT_HIVEMQ_USERNAME: str = _env("MQTT_HIVEMQ_USERNAME", "")
    MQTT_HIVEMQ_PASSWORD: str = _env("MQTT_HIVEMQ_PASSWORD", "")
    TOPIC_STATION2: str = _env("MQTT_TOPIC_STATION2", "factory/station02/telemetry")
    TOPIC_PROCESSED: str = _env("MQTT_TOPIC_PROCESSED", "factory/processed/telemetry")
    TOPIC_PREDICTIONS: str = _env("MQTT_TOPIC_PREDICTIONS", "factory/predictions")


config = AppConfig()

FEATURE_COLUMNS = [
    "rpm",
    "cycle_time",
    "throughput",
    "utilization",
    "queue_length",
    "vibration",
    "temperature",
    "current",
    "demand_rate",
    "line_throughput",
    "total_wip",
    "rpm_rolling_mean",
    "rpm_rolling_std",
    "rpm_trend",
    "tp_trend",
    "queue_growth_rate",
    "tp_to_demand_ratio",
]
TARGET_COLUMN = "target_bottleneck_60s"
MODEL_VERSION = "rf_v1.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 2. DATA SCHEMAS (plain dataclasses - no external validation library needed)
# =============================================================================
VALID_STATUSES = {"NORMAL", "WARNING", "CRITICAL", "STOPPED", "IDLE"}


@dataclass
class StationTelemetry:
    station_id: str
    station_name: str
    rpm: float
    cycle_time: float
    throughput: float
    utilization: float
    queue_length: int
    status: str
    source: str = "simulated"
    vibration: Optional[float] = None
    temperature: Optional[float] = None
    current: Optional[float] = None
    timestamp: str = field(default_factory=utc_now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        self.status = self.status.upper()
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}; must be one of {VALID_STATUSES}")
        if self.rpm < 0 or self.queue_length < 0:
            raise ValueError("rpm and queue_length must be >= 0")

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedLineTelemetry:
    demand_rate: float
    line_throughput: float
    total_wip: int
    line_efficiency: float
    bottleneck_station: Optional[str] = None
    stations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionMessage:
    bottleneck_predicted: bool
    risk_probability: float
    confidence: float
    predicted_station: Optional[str] = None
    prediction_horizon_sec: int = config.ML_PREDICTION_HORIZON_SEC
    model_version: str = MODEL_VERSION
    features_used: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 3. FACTORY SIMULATOR
# =============================================================================
class Workstation:
    """One industrial workstation on the production line."""

    def __init__(
        self,
        station_id: str,
        station_name: str,
        base_rpm: float,
        nominal_cycle_time: float,
        nominal_capacity: float,
        source: str = "simulated",
    ):
        self.station_id = station_id
        self.station_name = station_name
        self.base_rpm = base_rpm
        self.nominal_cycle_time = nominal_cycle_time
        self.nominal_capacity = nominal_capacity  # units / min
        self.source = source

        self.current_rpm = base_rpm
        self.target_rpm = base_rpm
        self.queue_length = 5
        self.current_throughput = nominal_capacity
        self.current_cycle_time = nominal_cycle_time
        self.utilization = 0.85
        self.status = "NORMAL"

        self.temperature = 36.5  # deg C
        self.vibration = 1.2  # mm/s RMS
        self.current = 0.85  # A
        self.hardware_active = False

    def update_hardware_reading(self, rpm: float, current=None, temp=None, vib=None):
        """Override simulated values with real ESP32 telemetry."""
        self.hardware_active = True
        self.source = "hardware"
        self.current_rpm = max(0.0, rpm)
        if current is not None:
            self.current = current
        if temp is not None:
            self.temperature = temp
        if vib is not None:
            self.vibration = vib

    def step(self, input_units: float, dt_sec: float) -> float:
        """Advance by dt_sec seconds; returns completed units."""
        self.queue_length = max(0, min(config.MAX_QUEUE_CAPACITY, self.queue_length + int(round(input_units))))

        if not self.hardware_active:
            if abs(self.current_rpm - self.target_rpm) > 1.0:
                self.current_rpm += (self.target_rpm - self.current_rpm) * min(1.0, dt_sec * 0.5)
            if self.current_rpm > 10.0:
                self.current_rpm = max(0.0, self.current_rpm + random.gauss(0, 1.5))

        rpm_ratio = self.current_rpm / max(1.0, self.base_rpm)
        effective_capacity = self.nominal_capacity * max(0.0, rpm_ratio)  # units / min
        max_possible_units = (effective_capacity / 60.0) * dt_sec

        if self.queue_length > 0 and effective_capacity > 0.05:
            units_processed = min(float(self.queue_length), max_possible_units)
            self.queue_length = max(0, int(round(self.queue_length - units_processed)))
            self.current_cycle_time = 60.0 / max(0.1, effective_capacity)
            self.utilization = min(1.0, max(0.1, effective_capacity / max(0.1, self.nominal_capacity)))
        else:
            units_processed = 0.0
            self.current_cycle_time = 99.9 if self.current_rpm < 10 else self.nominal_cycle_time
            self.utilization = 0.0 if self.current_rpm < 10 else 0.1

        if dt_sec > 0:
            instant_tp = (units_processed / dt_sec) * 60.0
            alpha = 0.15  # exponential moving average
            self.current_throughput = (1 - alpha) * self.current_throughput + alpha * instant_tp

        # Sensor behaviour depends on stress level
        if self.current_rpm < 50.0:
            self.status = "STOPPED"
            self.current = 0.0
            self.vibration = 0.05
        elif (
            self.queue_length >= config.ALERT_QUEUE_CRITICAL_THRESHOLD
            or rpm_ratio <= config.ALERT_RPM_CRITICAL_RATIO
        ):
            self.status = "CRITICAL"
            self.vibration = min(8.0, 3.5 + random.uniform(0.5, 2.0))
            self.temperature = min(75.0, self.temperature + 0.15 * dt_sec)
            self.current = min(3.0, 1.8 + random.uniform(0.2, 0.6))
        elif (
            self.queue_length >= config.ALERT_QUEUE_WARNING_THRESHOLD
            or rpm_ratio <= config.ALERT_RPM_WARNING_RATIO
        ):
            self.status = "WARNING"
            self.vibration = min(4.5, 2.0 + random.uniform(0.2, 0.8))
            self.temperature = min(55.0, self.temperature + 0.05 * dt_sec)
            self.current = min(2.0, 1.2 + random.uniform(0.1, 0.3))
        else:
            self.status = "NORMAL"
            self.vibration = max(0.8, 1.2 + random.uniform(-0.2, 0.2))
            if self.temperature > 37.0:
                self.temperature = max(35.0, self.temperature - 0.05 * dt_sec)
            else:
                self.temperature = 36.5 + random.uniform(-0.5, 0.5)
            self.current = max(0.6, 0.85 + random.uniform(-0.05, 0.05))

        return units_processed

    def to_telemetry(self) -> StationTelemetry:
        return StationTelemetry(
            station_id=self.station_id,
            station_name=self.station_name,
            source=self.source,
            rpm=round(self.current_rpm, 1),
            cycle_time=round(self.current_cycle_time, 2),
            throughput=round(max(0.0, self.current_throughput), 2),
            utilization=round(min(1.0, max(0.0, self.utilization)), 3),
            queue_length=int(self.queue_length),
            status=self.status,
            vibration=round(self.vibration, 2),
            temperature=round(self.temperature, 1),
            current=round(self.current, 2),
        )


class FactorySimulator:
    """3-station production line with several operating scenarios."""

    SCENARIOS = {
        "scenario_a": "Scenario A: Normal Factory Operation",
        "scenario_b": "Scenario B: Gradual Bottleneck Development",
        "scenario_c": "Scenario C: Sudden Machine Stoppage",
        "scenario_d": "Scenario D: High Production Demand",
        "scenario_e": "Scenario E: Recovery to Nominal Speed",
        "scenario_f": "Scenario F: Real ESP32 Hardware Integration",
    }

    def __init__(self):
        self.active_scenario = "scenario_a"
        self.scenario_elapsed = 0.0  # simulated seconds since scenario start
        self.time_scale = config.SIMULATION_TIME_SCALE
        self.station1 = Workstation(
            "station01", "Cutting", config.STATION1_BASE_RPM,
            config.STATION1_CYCLE_TIME_SEC, config.STATION1_CAPACITY_PER_MIN,
        )
        self.station2 = Workstation(
            "station02", "Drilling/Machining", config.STATION2_BASE_RPM,
            config.STATION2_CYCLE_TIME_SEC, config.STATION2_CAPACITY_PER_MIN,
        )
        self.station3 = Workstation(
            "station03", "Packaging", config.STATION3_BASE_RPM,
            config.STATION3_CYCLE_TIME_SEC, config.STATION3_CAPACITY_PER_MIN,
        )
        self.demand_rate = config.DEMAND_RATE_UNITS_PER_MIN
        self.last_step_time = time.time()
        self.total_finished_goods = 0

    def set_scenario(self, key: str):
        if key not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {key}")
        self.active_scenario = key
        self.scenario_elapsed = 0.0
        s2 = self.station2

        if key in ("scenario_a", "scenario_e"):
            s2.hardware_active = False
            s2.source = "simulated"
            s2.target_rpm = config.STATION2_BASE_RPM
            self.demand_rate = config.DEMAND_RATE_UNITS_PER_MIN
        elif key == "scenario_c":
            s2.hardware_active = False
            s2.source = "simulated"
            s2.target_rpm = 0.0
            s2.current_rpm = 0.0
        elif key == "scenario_d":
            s2.hardware_active = False
            s2.source = "simulated"
            s2.target_rpm = config.STATION2_BASE_RPM
            self.demand_rate = 24.0  # exceeds station capacity
        elif key == "scenario_f":
            s2.hardware_active = True
            s2.source = "hardware"

    def reset(self):
        for st, base, q in (
            (self.station1, config.STATION1_BASE_RPM, 5),
            (self.station2, config.STATION2_BASE_RPM, 6),
            (self.station3, config.STATION3_BASE_RPM, 4),
        ):
            st.queue_length = q
            st.current_rpm = base
            st.target_rpm = base
        self.station2.hardware_active = False
        self.station2.source = "simulated"
        self.demand_rate = config.DEMAND_RATE_UNITS_PER_MIN
        self.active_scenario = "scenario_a"
        self.scenario_elapsed = 0.0
        self.total_finished_goods = 0

    def step(self, dt_override: Optional[float] = None) -> Dict[str, Any]:
        now = time.time()
        dt = dt_override if dt_override is not None else (now - self.last_step_time)
        self.last_step_time = now

        effective_dt = max(0.1, min(5.0, dt * self.time_scale))
        self.scenario_elapsed += effective_dt  # simulated clock (fix)

        if self.active_scenario == "scenario_b":
            # Gradual slowdown: RPM falls to 25% of base over 90 simulated seconds
            progress = min(1.0, self.scenario_elapsed / 90.0)
            self.station2.target_rpm = config.STATION2_BASE_RPM * (1.0 - 0.75 * progress)
        elif self.active_scenario == "scenario_c":
            self.station2.target_rpm = 0.0
        elif self.active_scenario == "scenario_e":
            self.station2.target_rpm = config.STATION2_BASE_RPM

        incoming = (self.demand_rate / 60.0) * effective_dt
        st1_out = self.station1.step(incoming, effective_dt)
        st2_out = self.station2.step(st1_out, effective_dt)
        st3_out = self.station3.step(st2_out, effective_dt)
        self.total_finished_goods += int(round(st3_out))

        stations = [self.station1, self.station2, self.station3]
        candidate = max(stations, key=lambda s: s.queue_length)
        if candidate.queue_length < 8:
            stopped = [s for s in stations if s.status == "STOPPED"]
            bottleneck_id = stopped[0].station_id if stopped else None
        else:
            bottleneck_id = candidate.station_id

        t1, t2, t3 = (s.to_telemetry() for s in stations)
        total_wip = sum(s.queue_length for s in stations)
        line_tp = self.station3.current_throughput
        processed = ProcessedLineTelemetry(
            demand_rate=round(self.demand_rate, 1),
            line_throughput=round(line_tp, 2),
            total_wip=total_wip,
            bottleneck_station=bottleneck_id,
            line_efficiency=round(min(1.0, line_tp / max(1.0, self.demand_rate)), 3),
            stations={"station01": t1.model_dump(), "station02": t2.model_dump(), "station03": t3.model_dump()},
        )
        return {"station01": t1, "station02": t2, "station03": t3, "processed": processed}


# =============================================================================
# 4. SHARED FEATURE ENGINEERING (identical for training and live inference)
# =============================================================================
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised features for a whole run. Uses only samples <= t (no leakage)."""
    w, lag = config.ML_ROLLING_WINDOW_SIZE, config.ML_TREND_LAG
    df["rpm_rolling_mean"] = df["rpm"].rolling(window=w, min_periods=1).mean()
    df["rpm_rolling_std"] = df["rpm"].rolling(window=w, min_periods=1).std().fillna(0.0)
    df["rpm_trend"] = df["rpm"].diff(periods=lag).fillna(0.0) / lag
    df["tp_trend"] = df["throughput"].diff(periods=lag).fillna(0.0) / lag
    df["queue_growth_rate"] = df["queue_length"].diff(periods=lag).fillna(0.0) / lag
    df["tp_to_demand_ratio"] = df["throughput"] / np.maximum(1.0, df["demand_rate"])
    return df


def rolling_features_from_history(rpms: List[float], tps: List[float], queues: List[float]) -> Dict[str, float]:
    """Single-point version of add_rolling_features for streaming data (same math)."""
    w, lag = config.ML_ROLLING_WINDOW_SIZE, config.ML_TREND_LAG
    window = np.asarray(rpms[-w:], dtype=float)
    return {
        "rpm_rolling_mean": float(window.mean()),
        "rpm_rolling_std": float(window.std(ddof=1)) if len(window) > 1 else 0.0,  # pandas default ddof=1
        "rpm_trend": (rpms[-1] - rpms[-1 - lag]) / lag if len(rpms) > lag else 0.0,
        "tp_trend": (tps[-1] - tps[-1 - lag]) / lag if len(tps) > lag else 0.0,
        "queue_growth_rate": (queues[-1] - queues[-1 - lag]) / lag if len(queues) > lag else 0.0,
    }


# =============================================================================
# 5. DATASET GENERATION
# =============================================================================
def generate_simulation_runs(num_runs: int = 12, duration_sec: int = 300) -> pd.DataFrame:
    """Independent factory runs with varied scenarios and a 60-second look-ahead label."""
    logger.info("Generating %d simulation runs of %d seconds each...", num_runs, duration_sec)
    scenarios = ["scenario_a", "scenario_b", "scenario_c", "scenario_d", "scenario_e"]
    frames = []

    for run_id in range(1, num_runs + 1):
        scenario = random.choice(scenarios)
        sim = FactorySimulator()
        sim.time_scale = 1.0
        sim.set_scenario(scenario)
        sim.demand_rate = random.uniform(13.0, 18.0) if scenario != "scenario_d" else random.uniform(21.0, 26.0)

        dt, sim_time, records = 1.0, 0.0, []
        while sim_time < duration_sec:
            if sim_time == 120 and scenario == "scenario_a" and random.random() < 0.5:
                sim.set_scenario("scenario_b")
            elif sim_time == 180 and scenario == "scenario_b" and random.random() < 0.4:
                sim.set_scenario("scenario_e")

            step = sim.step(dt_override=dt)
            st1, st2, st3, line = step["station01"], step["station02"], step["station03"], step["processed"]
            records.append({
                "run_id": run_id,
                "sim_time": sim_time,
                "scenario": sim.active_scenario,
                "rpm": st2.rpm,
                "cycle_time": st2.cycle_time,
                "throughput": st2.throughput,
                "utilization": st2.utilization,
                "queue_length": st2.queue_length,
                "status": st2.status,
                "vibration": st2.vibration,
                "temperature": st2.temperature,
                "current": st2.current,
                "demand_rate": line.demand_rate,
                "line_throughput": line.line_throughput,
                "total_wip": line.total_wip,
                "st1_queue": st1.queue_length,
                "st1_throughput": st1.throughput,
                "st3_queue": st3.queue_length,
            })
            sim_time += dt

        run_df = add_rolling_features(pd.DataFrame(records))

        # Target: bottleneck somewhere in [t+10s, t+60s]
        # (future queue >= warning threshold OR future RPM < 900)
        horizon = config.ML_PREDICTION_HORIZON_SEC
        n = len(run_df)
        flags: List[float] = []
        for i in range(n):
            start, end = min(n, i + 10), min(n, i + horizon + 1)
            if start >= n or (end - start) < 10:
                flags.append(np.nan)  # end of run: label not computable
            else:
                q_max = run_df["queue_length"].iloc[start:end].max()
                r_min = run_df["rpm"].iloc[start:end].min()
                flags.append(int(q_max >= config.ALERT_QUEUE_WARNING_THRESHOLD or r_min < 900.0))
        run_df[TARGET_COLUMN] = flags
        run_df = run_df.dropna(subset=[TARGET_COLUMN]).copy()
        run_df[TARGET_COLUMN] = run_df[TARGET_COLUMN].astype(int)
        frames.append(run_df)

    return pd.concat(frames, ignore_index=True)


def cmd_generate(num_runs: int = 15, duration_sec: int = 300) -> pd.DataFrame:
    df = generate_simulation_runs(num_runs, duration_sec)
    config.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PATH, index=False)
    pos = int((df[TARGET_COLUMN] == 1).sum())
    neg = int((df[TARGET_COLUMN] == 0).sum())
    logger.info("Saved %d samples to %s", len(df), config.DATASET_PATH)
    logger.info("Class balance -> Bottleneck: %d (%.1f%%) | Normal: %d (%.1f%%)",
                pos, 100 * pos / len(df), neg, 100 * neg / len(df))
    return df


# =============================================================================
# 6. TRAINING
# =============================================================================
class RuleBasedBaseline:
    """Simple industrial heuristic used as a comparison baseline."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return ((X["queue_length"] >= 10) | (X["rpm_rolling_mean"] < 1100.0)).astype(int).values


def cmd_train(seed: int = 42) -> None:
    if not config.DATASET_PATH.exists():
        logger.info("No dataset found - generating one first.")
        cmd_generate()
    df = pd.read_csv(config.DATASET_PATH)

    # Episode-wise split: whole simulation runs go to train OR test (no leakage)
    run_ids = df["run_id"].unique()
    rng = np.random.RandomState(seed)
    n_test = max(2, int(len(run_ids) * 0.2))
    test_runs = set(rng.choice(run_ids, size=n_test, replace=False))
    train_runs = [r for r in run_ids if r not in test_runs]
    train_df = df[df["run_id"].isin(train_runs)]
    test_df = df[df["run_id"].isin(test_runs)]
    logger.info("Train: %d samples (%d runs) | Test: %d samples (%d runs)",
                len(train_df), len(train_runs), len(test_df), len(test_runs))

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    base_preds = RuleBasedBaseline().predict(X_test)
    base_f1 = f1_score(y_test, base_preds, zero_division=0)
    base_acc = accuracy_score(y_test, base_preds)
    logger.info("Baseline rule model -> accuracy %.3f, F1 %.3f", base_acc, base_f1)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1] if len(rf.classes_) > 1 else None
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    auc = float(roc_auc_score(y_test, probs)) if probs is not None and y_test.nunique() > 1 else None
    cm = confusion_matrix(y_test, preds).tolist()

    logger.info("=== Random Forest on unseen test runs ===")
    logger.info("Accuracy %.4f | Precision %.4f | Recall %.4f | F1 %.4f | ROC-AUC %s",
                acc, prec, rec, f1, f"{auc:.4f}" if auc is not None else "n/a")
    logger.info("Confusion matrix: %s", cm)

    importances = {c: float(v) for c, v in zip(FEATURE_COLUMNS, rf.feature_importances_)}
    for feat, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        logger.info("  top feature %-20s %.4f", feat, imp)

    config.ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": rf,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "trained_at": utc_now(),
        "version": MODEL_VERSION,
    }, config.ML_MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "trained_at": utc_now(),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "baseline_comparison": {
            "baseline_accuracy": round(base_acc, 4),
            "baseline_f1": round(base_f1, 4),
            "rf_f1_improvement": round(f1 - base_f1, 4),
        },
        "parameters": {
            "n_estimators": 100, "max_depth": 8,
            "prediction_horizon_sec": config.ML_PREDICTION_HORIZON_SEC,
            "class_weight": "balanced",
        },
    }
    config.ML_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Model saved to %s", config.ML_MODEL_PATH)


# =============================================================================
# 7. EVALUATION
# =============================================================================
def cmd_evaluate() -> None:
    if not config.ML_MODEL_PATH.exists():
        logger.error("Model not found at %s. Run `train` first.", config.ML_MODEL_PATH)
        return
    pkg = joblib.load(config.ML_MODEL_PATH)
    model, features = pkg["model"], pkg["feature_columns"]

    logger.info("Generating fresh evaluation data (5 unseen runs)...")
    eval_df = generate_simulation_runs(num_runs=5, duration_sec=200)
    X, y = eval_df[features], eval_df[TARGET_COLUMN]
    preds = model.predict(X)

    logger.info("=== Model %s benchmark on %d unseen samples ===", pkg.get("version", "?"), len(eval_df))
    logger.info("Accuracy: %.4f | F1: %.4f", accuracy_score(y, preds), f1_score(y, preds, zero_division=0))
    if y.nunique() > 1:
        logger.info("ROC-AUC: %.4f", roc_auc_score(y, model.predict_proba(X)[:, 1]))
    labels = sorted(y.unique())
    names = ["Normal", "Bottleneck"] if len(labels) == 2 else None
    print("\n" + classification_report(y, preds, labels=labels if names else None,
                                       target_names=names, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y, preds))


# =============================================================================
# 8. REAL-TIME PREDICTOR
# =============================================================================
class RealTimePredictor:
    """Streaming bottleneck inference: feed station-02 readings, get a risk score."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else config.ML_MODEL_PATH
        self.model = None
        self.feature_columns: List[str] = []
        self.model_version = "unavailable"
        self._load_model()

        self._lock = threading.Lock()
        keep = max(config.ML_ROLLING_WINDOW_SIZE, config.ML_TREND_LAG + 1)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=keep)
        self.last_line_metric: Dict[str, Any] = {
            "demand_rate": config.DEMAND_RATE_UNITS_PER_MIN,
            "line_throughput": 15.0,
            "total_wip": 15,
        }
        self._mqtt = None
        self.running = False

    def _load_model(self):
        if self.model_path.exists():
            try:
                pkg = joblib.load(self.model_path)
                self.model = pkg["model"]
                self.feature_columns = pkg["feature_columns"]
                self.model_version = pkg.get("version", MODEL_VERSION)
                logger.info("Loaded model '%s' from %s", self.model_version, self.model_path)
                return
            except Exception as exc:
                logger.error("Failed to load model: %s", exc)
                self.model_version = "corrupted_fallback"
        else:
            logger.warning("No model at %s - using rule-based fallback.", self.model_path)
            self.model_version = "rules_fallback"
        self.model = None

    def update_line_metrics(self, payload: Dict[str, Any]):
        """Feed processed line-level telemetry (demand, throughput, WIP)."""
        self.last_line_metric = payload

    def process_reading(self, reading: Dict[str, Any]) -> PredictionMessage:
        with self._lock:
            self.history.append(reading)
            buf = list(self.history)
        features = self._extract_features(buf, reading)

        if self.model is not None:
            try:
                feat_df = pd.DataFrame([features])[self.feature_columns]
                probs = self.model.predict_proba(feat_df)[0]
                classes = list(self.model.classes_)
                risk = float(probs[classes.index(1)]) if 1 in classes else 0.0
                is_bottleneck = risk >= 0.5
                confidence = float(max(probs))
            except Exception as exc:
                logger.error("Inference error: %s - using fallback rules.", exc)
                risk, is_bottleneck, confidence = self._rule_fallback(reading)
        else:
            risk, is_bottleneck, confidence = self._rule_fallback(reading)

        shown = ("rpm", "queue_length", "rpm_rolling_mean", "rpm_trend", "queue_growth_rate")
        return PredictionMessage(
            bottleneck_predicted=is_bottleneck,
            predicted_station="station02" if is_bottleneck else None,
            risk_probability=round(risk, 3),
            confidence=round(confidence, 3),
            model_version=self.model_version,
            features_used={k: round(v, 2) for k, v in features.items() if k in shown},
        )

    def _extract_features(self, history: List[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, float]:
        rpms = [float(r.get("rpm", 0.0)) for r in history]
        tps = [float(r.get("throughput", 0.0)) for r in history]
        queues = [float(r.get("queue_length", 0)) for r in history]
        rolling = rolling_features_from_history(rpms, tps, queues)

        demand = float(self.last_line_metric.get("demand_rate", config.DEMAND_RATE_UNITS_PER_MIN))
        curr_tp = float(current.get("throughput", 0.0))

        def val(key: str, default: float) -> float:
            v = current.get(key)
            return float(v) if v is not None else default

        return {
            "rpm": val("rpm", 0.0),
            "cycle_time": val("cycle_time", 4.0),
            "throughput": curr_tp,
            "utilization": val("utilization", 0.8),
            "queue_length": val("queue_length", 0.0),
            "vibration": val("vibration", 1.2),
            "temperature": val("temperature", 36.5),
            "current": val("current", 0.85),
            "demand_rate": demand,
            "line_throughput": float(self.last_line_metric.get("line_throughput", 15.0)),
            "total_wip": float(self.last_line_metric.get("total_wip", 15)),
            **rolling,
            "tp_to_demand_ratio": curr_tp / max(1.0, demand),
        }

    @staticmethod
    def _rule_fallback(reading: Dict[str, Any]) -> Tuple[float, bool, float]:
        rpm = float(reading.get("rpm", 1500.0))
        queue = int(reading.get("queue_length", 5))
        if rpm < 500 or queue >= 25:
            return 0.95, True, 0.90
        if rpm < 1000 or queue >= 15:
            return 0.70, True, 0.75
        if rpm < 1300 or queue >= 10:
            return 0.40, False, 0.65
        return 0.10, False, 0.90

    # ---- optional MQTT mode -------------------------------------------------
    def start_mqtt(self):
        """Subscribe to live telemetry, publish predictions. Requires paho-mqtt>=2.0."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise SystemExit("`live` mode needs paho-mqtt:  pip install paho-mqtt") from exc

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"factorypulse_predictor_{uuid.uuid4().hex[:6]}")
        use_cloud = config.MQTT_MODE == "hivemq"
        if use_cloud:
            client.username_pw_set(config.MQTT_HIVEMQ_USERNAME, config.MQTT_HIVEMQ_PASSWORD)
            client.tls_set()
            host, port = config.MQTT_HIVEMQ_HOST, config.MQTT_HIVEMQ_PORT
        else:
            host, port = config.MQTT_LOCAL_HOST, config.MQTT_LOCAL_PORT

        def on_connect(c, userdata, flags, reason_code, properties=None):
            logger.info("MQTT connected (%s). Subscribing...", reason_code)
            c.subscribe([(config.TOPIC_STATION2, 0), (config.TOPIC_PROCESSED, 0)])

        def on_message(c, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except ValueError:
                return
            if msg.topic == config.TOPIC_PROCESSED:
                self.update_line_metrics(payload)
            elif msg.topic == config.TOPIC_STATION2:
                pred = self.process_reading(payload)
                c.publish(config.TOPIC_PREDICTIONS, json.dumps(pred.model_dump()))
                logger.info("risk=%.2f bottleneck=%s", pred.risk_probability, pred.bottleneck_predicted)

        client.on_connect, client.on_message = on_connect, on_message
        client.connect(host, port, keepalive=60)
        self._mqtt = client
        self.running = True
        try:
            client.loop_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            client.disconnect()


# =============================================================================
# 9. DEMO (simulator -> predictor, no broker needed)
# =============================================================================
def cmd_demo(scenario: str = "scenario_b", steps: int = 150) -> None:
    predictor = RealTimePredictor()
    sim = FactorySimulator()
    sim.set_scenario(scenario)
    print(f"\nDemo: {FactorySimulator.SCENARIOS[scenario]}  |  model: {predictor.model_version}")
    print(f"{'t(s)':>5} {'RPM':>8} {'queue':>6} {'status':>9} {'risk':>6}  prediction (next {config.ML_PREDICTION_HORIZON_SEC}s)")
    for t in range(steps):
        out = sim.step(dt_override=1.0)
        predictor.update_line_metrics(out["processed"].model_dump())
        s2 = out["station02"]
        pred = predictor.process_reading(s2.model_dump())
        if t % 10 == 0:
            flag = "BOTTLENECK LIKELY" if pred.bottleneck_predicted else "ok"
            print(f"{t:>5} {s2.rpm:>8.1f} {s2.queue_length:>6} {s2.status:>9} {pred.risk_probability:>6.2f}  {flag}")


# =============================================================================
# 10. COMMAND LINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FactoryPulse AI - bottleneck prediction (single file)")
    parser.add_argument("command", choices=["generate", "train", "evaluate", "demo", "live", "all"])
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    parser.add_argument("--runs", type=int, default=15, help="simulation runs for `generate`")
    parser.add_argument("--duration", type=int, default=300, help="seconds per run for `generate`")
    parser.add_argument("--scenario", default="scenario_b", choices=list(FactorySimulator.SCENARIOS),
                        help="scenario for `demo`")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    train_seed = args.seed if args.seed is not None else 42
    if args.command == "generate":
        cmd_generate(args.runs, args.duration)
    elif args.command == "train":
        cmd_train(train_seed)
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "demo":
        cmd_demo(args.scenario)
    elif args.command == "live":
        RealTimePredictor().start_mqtt()
    elif args.command == "all":
        cmd_generate(args.runs, args.duration)
        cmd_train(train_seed)
        cmd_evaluate()
        cmd_demo(args.scenario)


if __name__ == "__main__":
    main()
