"""
Unit Tests for Controllers, FSM, Safety Filters, and Gymnasium Environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from anesthesionet.physiology import PatientProfile, PatientSimulator
from anesthesionet.environment import AnesthesiaClosedLoopEnv
from anesthesionet.controllers import InductionController, FullLifecycleController, SurgicalStage
from anesthesionet.safety import HOCBFSafetyFilter


def test_gym_environment():
    patient = PatientProfile(age_years=40, weight_kg=70, height_cm=175, sex="male")
    env = AnesthesiaClosedLoopEnv(patient=patient, control_interval_sec=1.0)
    
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert 90.0 <= obs[0] <= 100.0  # Awake BIS
    
    # Step with moderate dose
    action = np.array([4.0, 0.2, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    
    assert next_obs.shape == (8,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated


def test_induction_hysteresis_lock():
    patient = PatientProfile(age_years=45, weight_kg=70, height_cm=175, sex="male")
    controller = InductionController(patient)
    sim = PatientSimulator(patient)
    
    bolus = controller.start_induction()
    assert bolus > 100.0  # Valid induction bolus
    assert controller.state == "TCI_RAMPING"
    sim.administer_propofol_bolus(bolus)
    
    # Simulate steps until clearance
    cleared = False
    for _ in range(300):
        v = sim.step(propofol_rate_mg_kg_h=5.0, remifentanil_rate_ug_kg_min=0.2, dt_seconds=1.0)
        p, r, n, is_cleared = controller.compute_infusion_rates(v, dt_seconds=1.0)
        if is_cleared:
            cleared = True
            break
            
    assert cleared
    assert controller.state == "CLEARANCE_APPROVED"
    assert controller.consecutive_dwell_sec >= 30.0


def test_hocbf_safety_filter():
    filter_engine = HOCBFSafetyFilter()
    patient = PatientProfile(age_years=50, weight_kg=70, height_cm=175, sex="male")
    sim = PatientSimulator(patient)
    
    # Step simulator to a deeply anesthetized, hypotensive state
    sim.pk_prop.state[0] = 30.0   # High propofol mass
    sim.hemodynamics.current_map = 62.0  # Hypotension breach!
    sim.hemodynamics.current_hr = 60.0
    vitals = sim.step(dt_seconds=0.1)
    
    # Controller proposes dangerously high propofol and zero pressor
    dangerous_action = np.array([10.0, 0.5, 0.0], dtype=np.float32)
    safe_action, was_intervened, barrier_vals = filter_engine.filter_action(dangerous_action, vitals)
    
    assert was_intervened
    # Pressor should be augmented and propofol throttled
    assert safe_action[2] > 0.0 or safe_action[0] < dangerous_action[0]


def test_full_lifecycle_flow():
    patient = PatientProfile(age_years=35, weight_kg=65, height_cm=170, sex="female")
    fsm = FullLifecycleController(patient)
    sim = PatientSimulator(patient)
    
    # 1. Trigger Induction
    bolus = fsm.trigger_induction()
    sim.administer_propofol_bolus(bolus)
    assert fsm.stage == SurgicalStage.STAGE_1_INDUCTION
    
    # Step for 150 seconds to progress through hysteresis to maintenance
    vitals = sim.step(dt_seconds=0.1)
    for _ in range(150):
        action, _, info = fsm.step(vitals, dt_seconds=1.0)
        vitals = sim.step(
            propofol_rate_mg_kg_h=float(action[0]),
            remifentanil_rate_ug_kg_min=float(action[1]),
            norepinephrine_rate_ug_kg_min=float(action[2]),
            dt_seconds=1.0
        )
        
    assert fsm.stage in [SurgicalStage.READINESS_HYSTERESIS, SurgicalStage.STAGE_2_MAINTENANCE]


if __name__ == "__main__":
    test_gym_environment()
    test_induction_hysteresis_lock()
    test_hocbf_safety_filter()
    test_full_lifecycle_flow()
    print("All Step 2 controller & safety unit tests passed successfully!")
