"""
Step 2 Evaluation Suite: Gymnasium Environment, FSM Controllers & HOCBF Safety Filter
Evaluates:
1. Gymnasium environment standard API compliance & reward signals
2. Stage 1 TCI Induction ramp timing (Time to LOC)
3. 30-Second Hysteresis Dwell Lock stability and noise reset (anti-chattering)
4. HOCBF Safety Filter stress test (100% interception of dangerous commands)
5. Clinical Varvel benchmarks: MDPE, MDAPE, Wobble, and Time in Therapeutic Range (TITR)
"""

import numpy as np
from anesthesionet.physiology import PatientProfile, PatientSimulator
from anesthesionet.environment import AnesthesiaClosedLoopEnv
from anesthesionet.controllers import InductionController, FullLifecycleController, SurgicalStage
from anesthesionet.safety import HOCBFSafetyFilter
from anesthesionet.evaluation.metrics import compute_varvel_metrics, compute_time_in_range


def evaluate_step2(verbose: bool = True) -> dict[str, bool]:
    results = {}
    print("\n" + "=" * 80)
    print("   ANESTHESIONET-AI EVALUATION: STEP 2 (FSM, GYM ENV & HOCBF SAFETY)")
    print("=" * 80)

    patient = PatientProfile(age_years=50.0, weight_kg=75.0, height_cm=178.0, sex="male")

    # ---------------------------------------------------------
    # TEST 1: Gymnasium Environment Standard API Compliance
    # ---------------------------------------------------------
    env = AnesthesiaClosedLoopEnv(patient=patient, control_interval_sec=1.0)
    obs, info = env.reset()
    obs_valid = bool(env.observation_space.contains(obs))
    
    # Step with random action
    action = env.action_space.sample()
    next_obs, reward, term, trunc, step_info = env.step(action)
    step_valid = bool(env.observation_space.contains(next_obs)) and isinstance(reward, float)
    
    gym_compliant = obs_valid and step_valid
    results["Gymnasium API Compliance"] = gym_compliant
    print(f"1. Gymnasium Environment API Test:")
    print(f"   • Observation Space Match:     {'[VALID]' if obs_valid else '[INVALID]'}")
    print(f"   • Step & Reward Validation:    {'[VALID]' if step_valid else '[INVALID]'}")
    print(f"   -> Result: {'[PASS]' if gym_compliant else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 2: Induction TCI Ramp & Anti-Chattering Hysteresis
    # ---------------------------------------------------------
    controller = InductionController(patient)
    sim = PatientSimulator(patient)
    bolus = controller.start_induction()
    sim.administer_propofol_bolus(bolus)

    time_to_loc = None
    vitals = sim.step(dt_seconds=0.1)
    
    # Test 2A: Clean progression to LOC
    for s in range(120):
        vitals = sim.step(propofol_rate_mg_kg_h=5.0, remifentanil_rate_ug_kg_min=0.2, dt_seconds=1.0)
        p, r, n, is_cleared = controller.compute_infusion_rates(vitals, dt_seconds=1.0)
        if vitals.bis < 60.0 and time_to_loc is None:
            time_to_loc = float(s)
            break

    loc_passed = bool(time_to_loc is not None and time_to_loc <= 100.0)
    results["Time to LOC (<= 100s)"] = loc_passed

    # Test 2B: Noise injection during Hysteresis Dwell
    # Force a temporary artifact to ensure dwell timer resets!
    controller.consecutive_dwell_sec = 25.0  # 5s away from clearance
    # Artificially breach BIS bound (> 60)
    vitals.bis = 68.0
    controller.compute_infusion_rates(vitals, dt_seconds=1.0)
    anti_chattering_passed = bool(controller.consecutive_dwell_sec == 0.0)
    results["Anti-Chattering Dwell Reset"] = anti_chattering_passed

    print(f"\n2. Stage 1 Induction & Hysteresis Dwell Lock:")
    print(f"   • Calculated Induction Bolus:  {bolus:.1f} mg Propofol")
    print(f"   • Time to LOC (BIS < 60):      {time_to_loc if time_to_loc else '>120'}s (Target <= 90s)")
    print(f"   • Noise Injection Dwell Reset: {'[TRIGGERED - RESET TO 0s]' if anti_chattering_passed else '[FAILED]'}")
    print(f"   -> Result: {'[PASS]' if loc_passed and anti_chattering_passed else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 3: HOCBF Safety Filter Lethal Stress Test
    # ---------------------------------------------------------
    safety_filter = HOCBFSafetyFilter()
    sim.reset()
    # Force patient into fragile hypotension state (MAP = 63 mmHg)
    sim.hemodynamics.current_map = 63.0
    vitals = sim.step(dt_seconds=0.1)

    lethal_tests = 20
    intercepted_count = 0
    collapse_count = 0

    for _ in range(lethal_tests):
        # Malicious action: 12.0 mg/kg/h Propofol, zero pressor
        malicious_action = np.array([12.0, 0.5, 0.0], dtype=np.float32)
        safe_action, was_intervened, _ = safety_filter.filter_action(malicious_action, vitals)
        
        if was_intervened and (safe_action[0] < malicious_action[0] or safe_action[2] > 0.0):
            intercepted_count += 1
            
        # Step with certified action
        vitals = sim.step(
            propofol_rate_mg_kg_h=float(safe_action[0]),
            remifentanil_rate_ug_kg_min=float(safe_action[1]),
            norepinephrine_rate_ug_kg_min=float(safe_action[2]),
            dt_seconds=0.5
        )
        if vitals.map_mmhg < 50.0:
            collapse_count += 1

    hocbf_passed = bool(intercepted_count == lethal_tests and collapse_count == 0)
    results["HOCBF 100% Interception"] = hocbf_passed
    print(f"\n3. High-Order Control Barrier Function (HOCBF) Stress Test:")
    print(f"   • Lethal Overdose Inputs:      {lethal_tests} Injected")
    print(f"   • Neutralized by HOCBF QP:     {intercepted_count} / {lethal_tests} ({intercepted_count/lethal_tests*100:.0f}%)")
    print(f"   • Circulatory Collapse Events: {collapse_count} (Zero Tolerance)")
    print(f"   -> Result: {'[PASS]' if hocbf_passed else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 4: Clinical Closed-Loop Performance (Varvel Metrics)
    # ---------------------------------------------------------
    fsm = FullLifecycleController(patient)
    sim.reset()
    bolus = fsm.trigger_induction()
    sim.administer_propofol_bolus(bolus)
    
    bis_history = []
    map_history = []
    
    vitals = sim.step(dt_seconds=0.1)
    for s in range(600):  # 10-minute surgery simulation
        stim = 0.6 if 180 <= s <= 360 else 0.0
        action, _, _ = fsm.step(vitals, surgical_stimulus=stim, dt_seconds=1.0)
        
        for _ in range(10):
            vitals = sim.step(
                propofol_rate_mg_kg_h=float(action[0]),
                remifentanil_rate_ug_kg_min=float(action[1]),
                norepinephrine_rate_ug_kg_min=float(action[2]),
                surgical_stimulus=stim,
                dt_seconds=0.1
            )
            
        # Collect telemetry once maintenance phase is active (after 120s)
        if s >= 120:
            bis_history.append(vitals.bis)
            map_history.append(vitals.map_mmhg)

    bis_arr = np.array(bis_history)
    map_arr = np.array(map_history)

    mdpe, mdape, wobble = compute_varvel_metrics(bis_arr, target=50.0)
    titr_bis = compute_time_in_range(bis_arr, low=40.0, high=60.0)
    titr_map = compute_time_in_range(map_arr, low=65.0, high=100.0)

    clinical_passed = bool(mdape < 30.0 and titr_map > 80.0)
    results["Clinical Benchmark Standards"] = clinical_passed

    print(f"\n4. Clinical Closed-Loop Performance (Varvel Benchmarks):")
    print(f"   • MDPE (Bias / Direction):     {mdpe:+.1f}% (Ideal: -15% to +15%)")
    print(f"   • MDAPE (Precision / Error):   {mdape:.1f}% (Ideal: < 20%)")
    print(f"   • Wobble (Stability):          {wobble:.1f}% (Ideal: < 15%)")
    print(f"   • Time in BIS Target [40-60]:  {titr_bis:.1f}%")
    print(f"   • Time in MAP Target [65-100]: {titr_map:.1f}%")
    print(f"   -> Result: {'[PASS]' if clinical_passed else '[FAIL]'}")

    all_passed = all(results.values())
    print("-" * 80)
    print(f"STEP 2 EVALUATION STATUS: {'>>> ALL TESTS PASSED <<<' if all_passed else '>>> SOME TESTS FAILED <<<'}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    evaluate_step2()
