"""
Unit Tests for Signal Filter, Mode B VitalDB Replay, and Counterfactual Estimator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from anesthesionet.signal import SignalArtifactFilter
from anesthesionet.replay import (
    generate_benchmark_vitaldb_case,
    VitalDBReplayEngine,
    CounterfactualTransferEstimator
)
from anesthesionet.engine_selector import DualEngineCoordinator, ExecutionMode
from anesthesionet.physiology import PatientProfile


def test_velocity_gate_rejection():
    filt = SignalArtifactFilter()
    # Baseline
    filt.process(0.0, raw_bis=50.0, raw_map=80.0, raw_hr=70.0)
    # 40-unit spike in 1s
    out = filt.process(1.0, raw_bis=90.0, raw_map=80.0, raw_hr=70.0)
    assert out.is_artifact
    assert out.filtered_bis == 50.0  # Held state


def test_sqi_emg_detection():
    filt = SignalArtifactFilter()
    out = filt.process(0.0, raw_bis=50.0, raw_map=80.0, raw_hr=70.0, sqi_percent=35.0, emg_db=50.0)
    assert out.is_artifact


def test_sensor_hold_activation():
    filt = SignalArtifactFilter()
    # 4 consecutive seconds of noise
    for t in range(4):
        out = filt.process(float(t), raw_bis=95.0, raw_map=80.0, raw_hr=70.0, sqi_percent=20.0, emg_db=55.0)
    assert out.sensor_hold_active


def test_vitaldb_replay_playback():
    case = generate_benchmark_vitaldb_case(case_id=999, duration_seconds=100)
    engine = VitalDBReplayEngine(case)
    frames = list(engine.stream(time_scale=0.0))
    assert len(frames) == 100
    assert frames[0].time_seconds == 0.0
    assert frames[-1].time_seconds == 99.0


def test_counterfactual_estimator():
    patient = PatientProfile(age_years=45, weight_kg=70, height_cm=175, sex="male")
    cf = CounterfactualTransferEstimator(patient)
    
    # When actions match, counterfactual == dataset
    u = np.array([4.0, 0.2, 0.0], dtype=np.float32)
    proj = cf.evaluate_step(0.0, dataset_bis=50.0, dataset_map=75.0, doctor_action=u, ai_action=u)
    assert abs(proj.counterfactual_bis - 50.0) < 0.05
    assert abs(proj.counterfactual_map - 75.0) < 0.05


def test_dual_engine_switch():
    coordinator = DualEngineCoordinator()
    coordinator.switch_mode(ExecutionMode.MODE_B_REPLAY)
    assert coordinator.mode == ExecutionMode.MODE_B_REPLAY
    
    tel, cf, meta = coordinator.step(ai_action=np.array([4.0, 0.15, 0.0]))
    assert meta["mode"] == "MODE_B_REPLAY"
    assert cf is not None


if __name__ == "__main__":
    test_velocity_gate_rejection()
    test_sqi_emg_detection()
    test_sensor_hold_activation()
    test_vitaldb_replay_playback()
    test_counterfactual_estimator()
    test_dual_engine_switch()
    print("All Step 3 Signal & Replay unit tests passed successfully!")
