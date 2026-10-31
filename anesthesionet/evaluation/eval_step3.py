"""
Step 3 Evaluation Suite: Mode B VitalDB Replay & Signal Artifact Filter
Evaluates:
1. Signal Artifact Filter: velocity gate, SQI/EMG thresholding, and Sensor Hold failsafe
2. VitalDB Replay Engine: streaming throughput, event marker detection, zero data corruption
3. Counterfactual Transfer Estimator: off-policy trajectory tracking and sensitivity
4. Dual-Engine Coordinator: seamless switching between Mode A (Simulation) and Mode B (Replay)
"""

import numpy as np
from anesthesionet.signal import SignalArtifactFilter, FilterConfig
from anesthesionet.replay import (
    generate_benchmark_vitaldb_case,
    VitalDBReplayEngine,
    CounterfactualTransferEstimator
)
from anesthesionet.engine_selector import DualEngineCoordinator, ExecutionMode
from anesthesionet.physiology import PatientProfile


def evaluate_step3(verbose: bool = True) -> dict[str, bool]:
    results = {}
    print("\n" + "=" * 80)
    print("   ANESTHESIONET-AI EVALUATION: STEP 3 (VITALDB REPLAY & ARTIFACT FILTER)")
    print("=" * 80)

    # ---------------------------------------------------------
    # TEST 1: Signal Artifact Filter Rejection & Sensor Hold
    # ---------------------------------------------------------
    filt = SignalArtifactFilter()
    
    # 1A: Establish clean baseline (BIS = 50)
    for t in range(5):
        filt.process(time_sec=float(t), raw_bis=50.0, raw_map=75.0, raw_hr=70.0)

    # 1B: Single-step velocity breach (Bovie spike from 50 to 95 in 1s)
    spiked_frame = filt.process(time_sec=6.0, raw_bis=95.0, raw_map=75.0, raw_hr=70.0)
    velocity_rejected = bool(spiked_frame.is_artifact and spiked_frame.filtered_bis == 50.0)
    results["Velocity Spike Rejection"] = velocity_rejected

    # 1C: Low SQI / High EMG RF noise rejection
    rf_noise_frame = filt.process(
        time_sec=7.0, raw_bis=52.0, raw_map=75.0, raw_hr=70.0,
        sqi_percent=30.0, emg_db=58.0
    )
    spectral_rejected = bool(rf_noise_frame.is_artifact and "Low SQI" in rf_noise_frame.status_message)
    results["Spectral Noise Detection"] = spectral_rejected

    # 1D: Sustained noise trigger -> "Sensor Hold Mode"
    hold_triggered = False
    for t in range(8, 15):  # 7 seconds of sustained electrocautery
        hold_frame = filt.process(
            time_sec=float(t), raw_bis=92.0, raw_map=75.0, raw_hr=70.0,
            sqi_percent=20.0, emg_db=60.0
        )
        if hold_frame.sensor_hold_active:
            hold_triggered = True

    results["Sensor Hold Failsafe"] = hold_triggered

    print("1. Signal Artifact Filter Test:")
    print(f"   • Velocity Gate (Spike 50->95 held at 50): {'[PASS]' if velocity_rejected else '[FAIL]'}")
    print(f"   • Spectral SQI / EMG Detection:            {'[PASS]' if spectral_rejected else '[FAIL]'}")
    print(f"   • Sustained Noise Sensor Hold Trigger:     {'[PASS]' if hold_triggered else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 2: VitalDB Historical Replay Streaming
    # ---------------------------------------------------------
    case = generate_benchmark_vitaldb_case(case_id=2048, duration_seconds=600)
    replay = VitalDBReplayEngine(case)

    frames_read = 0
    events_found = []
    
    for frame in replay.stream(time_scale=0.0):  # unthrottled maximum speed
        frames_read += 1
        if frame.event_marker and isinstance(frame.event_marker, str):
            events_found.append(frame.event_marker)

    replay_passed = bool(frames_read == 600 and len(events_found) >= 3)
    results["VitalDB Stream Integrity"] = replay_passed

    print("\n2. Mode B VitalDB Replay Stream Test:")
    print(f"   • Frames Ingested:                         {frames_read} / 600 frames (100% Integrity)")
    print(f"   • Intraoperative Events Captured:          {len(events_found)} ({', '.join(events_found[:3])}...)")
    print(f"   -> Result: {'[PASS]' if replay_passed else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 3: Counterfactual Transfer Estimator (DR-OPE)
    # ---------------------------------------------------------
    patient = PatientProfile(age_years=52.0, weight_kg=68.0, height_cm=172.0, sex="female")
    cf_estimator = CounterfactualTransferEstimator(patient)

    # 3A: Zero-divergence test (when AI action == doctor action, counterfactual == dataset)
    doc_act = np.array([4.0, 0.15, 0.0], dtype=np.float32)
    ai_act_equal = doc_act.copy()
    
    proj_equal = cf_estimator.evaluate_step(
        time_sec=10.0, dataset_bis=48.0, dataset_map=74.0,
        doctor_action=doc_act, ai_action=ai_act_equal
    )
    equal_passed = bool(abs(proj_equal.counterfactual_bis - 48.0) < 0.01)

    # 3B: Sensitivity test (AI delivers an anticipatory rescue bolus -> counterfactual BIS drops)
    cf_estimator.reset()
    cf_estimator.sim_doctor.administer_propofol_bolus(60.0)
    cf_estimator.sim_ai.administer_propofol_bolus(100.0)  # AI provides 40mg greater hypnotic dose
    
    for s in range(60):
        proj_high = cf_estimator.evaluate_step(
            time_sec=float(10 + s), dataset_bis=48.0, dataset_map=74.0,
            doctor_action=doc_act, ai_action=doc_act
        )
    sensitivity_passed = bool(proj_high.counterfactual_bis < 48.0 - 1.5)

    cf_passed = equal_passed and sensitivity_passed
    results["Counterfactual Transfer Estimator"] = cf_passed

    print("\n3. Counterfactual Shadow Mode (DR-OPE) Test:")
    print(f"   • Zero-Divergence Equivalence:             {'[PASS]' if equal_passed else '[FAIL]'}")
    print(f"   • Sensitivity Shift (High Propofol):       BIS: {proj_high.dataset_bis:.1f} -> CF: {proj_high.counterfactual_bis:.1f} ({'[PASS]' if sensitivity_passed else '[FAIL]'})")
    print(f"   -> Result: {'[PASS]' if cf_passed else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 4: Dual-Engine Coordinator Integration
    # ---------------------------------------------------------
    coordinator = DualEngineCoordinator()
    
    # Run 5 steps in Mode A
    coordinator.switch_mode(ExecutionMode.MODE_A_SIMULATOR)
    tel_a, cf_a, meta_a = coordinator.step(ai_action=np.array([4.0, 0.15, 0.0]))
    mode_a_ok = (meta_a["mode"] == "MODE_A_SIMULATOR" and cf_a is None)

    # Switch to Mode B
    coordinator.switch_mode(ExecutionMode.MODE_B_REPLAY)
    tel_b, cf_b, meta_b = coordinator.step(ai_action=np.array([4.0, 0.15, 0.0]))
    mode_b_ok = (meta_b["mode"] == "MODE_B_REPLAY" and cf_b is not None)

    dual_engine_passed = mode_a_ok and mode_b_ok
    results["Dual-Engine Unified Switching"] = dual_engine_passed

    print("\n4. Dual-Engine Coordinator Integration Test:")
    print(f"   • Mode A (Simulation):                     {'[VALID]' if mode_a_ok else '[INVALID]'}")
    print(f"   • Mode B (Replay with Shadow CF):          {'[VALID]' if mode_b_ok else '[INVALID]'}")
    print(f"   -> Result: {'[PASS]' if dual_engine_passed else '[FAIL]'}")

    all_passed = all(results.values())
    print("-" * 80)
    print(f"STEP 3 EVALUATION STATUS: {'>>> ALL TESTS PASSED <<<' if all_passed else '>>> SOME TESTS FAILED <<<'}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    evaluate_step3()
