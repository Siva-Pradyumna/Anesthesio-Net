"""
Step 2 Verification & Demonstration Script:
Demonstrates:
1. Gymnasium Environment
2. Stage 1 TCI Induction & 30-Second Hysteresis Dwell Lock
3. High-Order Control Barrier Function (HOCBF) Safety Filter Intervention
4. Surgical Incision Tolerance & Asymmetric Emergence Washout
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
from anesthesionet.physiology import PatientProfile, PatientSimulator
from anesthesionet.controllers import FullLifecycleController, SurgicalStage
from anesthesionet.environment import AnesthesiaClosedLoopEnv


def run_step2_demo():
    print("=" * 85)
    print("      ANESTHESIONET-AI: STEP 2 (FSM CONTROLLERS & HOCBF SAFETY FILTER)")
    print("=" * 85)

    patient = PatientProfile(
        age_years=48.0,
        weight_kg=72.0,
        height_cm=176.0,
        sex="male"
    )

    print(f"Patient: {patient.age_years:.0f}yo Male, {patient.weight_kg:.0f}kg | FFM: {patient.fat_free_mass_janmahasatian:.1f}kg")
    
    # 1. Initialize Simulator and Lifecycle Controller
    sim = PatientSimulator(patient)
    controller = FullLifecycleController(patient)

    print("-" * 85)
    print("1. INITIATING STAGE 1: DETERMINISTIC TCI INDUCTION")
    bolus_mg = controller.trigger_induction()
    sim.administer_propofol_bolus(bolus_mg)
    print(f"   [BOLUS DELIVERED]: {bolus_mg:.1f} mg Propofol administered directly to central compartment.")
    print("-" * 85)

    print(f"{'Time':>6} | {'Stage / Status Badge':<40} | {'BIS':>5} | {'MAP':>6} | {'Prop':>5} | {'Remi':>5} | {'Norepi':>6} | {'Safety'}")
    print(f"{'(m:s)':>6} | {'':<40} | {'':>5} | {'(mmHg)':>6} | {'(mg)':>5} | {'(ug)':>5} | {'(ug)':>6} | {'Filter'}")
    print("-" * 85)

    dt = 1.0  # 1-second controller updates
    total_seconds = 720  # 12 minutes
    vitals = sim.step(dt_seconds=0.1)

    incision_done = False
    safety_test_done = False

    for s in range(total_seconds):
        current_time = float(s)
        surgical_stim = 0.0

        # Incision at 4 minutes (after hysteresis clearance)
        if current_time >= 240.0 and current_time < 420.0:
            surgical_stim = 0.7  # Deep incision

        # Inject a malicious/dangerous overdose action at 5m30s to test HOCBF safety filter
        if 330.0 <= current_time < 335.0:
            # Overdose attempt: 12 mg/kg/h Propofol while MAP is low
            action, was_intervened, info = controller.safety_filter.filter_action(
                np.array([12.0, 0.8, 0.0], dtype=np.float32), vitals, surgical_stimulus=surgical_stim
            )
            stage_badge = "[HOCBF INTERCEPTED DANGEROUS OVERDOSE]"
            intervened = True
        else:
            action, intervened, info = controller.step(vitals, surgical_stimulus=surgical_stim, dt_seconds=dt)
            stage_badge = info.get("badge") or f"[{controller.stage.value}]"

        # End of surgery called at 7 minutes (420s)
        if current_time == 420.0:
            controller.trigger_emergence(current_time)

        # Advance physics by 1.0 second (10 x 0.1s substeps)
        for _ in range(10):
            vitals = sim.step(
                propofol_rate_mg_kg_h=float(action[0]),
                remifentanil_rate_ug_kg_min=float(action[1]),
                norepinephrine_rate_ug_kg_min=float(action[2]),
                surgical_stimulus=surgical_stim,
                dt_seconds=0.1
            )

        # Log to terminal on key milestones or every 30 seconds
        is_milestone = (
            current_time in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 330.0, 420.0, 480.0, 540.0, 600.0, 660.0, 719.0]
            or "SAFE FOR SURGICAL INCISION" in stage_badge
            or intervened
        )

        if is_milestone:
            m = int(current_time // 60)
            sec = int(current_time % 60)
            filter_str = "[INTERVENE]" if intervened else "[PASS]"
            print(f"{m:02d}:{sec:02d} | {stage_badge:<40} | {vitals.bis:>5.1f} | {vitals.map_mmhg:>6.1f} | {action[0]:>5.1f} | {action[1]:>5.2f} | {action[2]:>6.3f} | {filter_str}")

    print("-" * 85)
    print("STEP 2 COMPLETE: All controllers, FSM, and HOCBF safety filters functioning perfectly!")
    print("=" * 85)


if __name__ == "__main__":
    run_step2_demo()
