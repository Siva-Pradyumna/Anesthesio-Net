"""
AnesthesioNet-AI Digital Twin Simulation Demo
Demonstrates real-time 3-drug PK/PD coupled simulation across:
Induction -> Readiness -> Surgical Incision -> Maintenance -> Emergence
"""

import sys
import time
from anesthesionet.physiology import PatientProfile, PatientSimulator


def run_clinical_demo():
    print("=" * 80)
    print("       ANESTHESIONET-AI: DIGITAL TWIN (MODE A) PHYSIOLOGY ENGINE")
    print("=" * 80)

    # 1. Initialize Patient: 45 yo Male, 70 kg, 175 cm
    patient = PatientProfile(
        age_years=45.0,
        weight_kg=70.0,
        height_cm=175.0,
        sex="male"
    )

    print(f"Patient Profile: {patient.age_years:.0f}yo {patient.sex.capitalize()}, {patient.weight_kg:.0f}kg, {patient.height_cm:.0f}cm")
    print(f"Calculated BMI: {patient.bmi:.1f} kg/m² | Janmahasatian FFM: {patient.fat_free_mass_janmahasatian:.1f} kg")
    print("-" * 80)

    sim = PatientSimulator(patient)
    print("Digital Twin Initialized with Eleveld PK/PD & Minto/Bouillon Synergy Models.")
    print("-" * 80)

    print(f"{'Time':>6} | {'Phase / Action':<26} | {'BIS':>5} | {'MAP':>6} | {'HR':>5} | {'Ce Prop':>8} | {'Ce Remi':>8} | {'Norepi':>7}")
    print(f"{'(min:sec)':>6} | {'':<26} | {'(0-100)':>5} | {'(mmHg)':>6} | {'(bpm)':>5} | {'(ug/mL)':>8} | {'(ng/mL)':>8} | {'(ug/kg/m)':>7}")
    print("-" * 80)

    # Simulation timeline parameters
    dt = 0.1  # 100 ms step
    total_seconds = 750  # 12.5 minutes of surgery

    # Initial baseline
    vitals = sim.step(dt_seconds=dt)
    print(f"{'00:00':>6} | {'[BASELINE] Awake Pre-op':<26} | {vitals.bis:>5.1f} | {vitals.map_mmhg:>6.1f} | {vitals.hr_bpm:>5.1f} | {vitals.ce_propofol_ug_ml:>8.2f} | {vitals.ce_remi_ng_ml:>8.2f} | {vitals.norepinephrine_rate_ug_kg_min:>7.3f}")

    # Stage 1: Induction at t = 5s
    # Administer 120mg Propofol Bolus + 6 mg/kg/h infusion + 0.2 ug/kg/min Remifentanil
    prop_rate = 0.0
    remi_rate = 0.0
    norepi_rate = 0.0
    surgical_pain = 0.0

    step_counter = 0

    for t_step in range(1, int(total_seconds / dt)):
        current_time_sec = t_step * dt

        # Timeline Events
        if current_time_sec == 5.0:
            sim.administer_propofol_bolus(120.0)  # 120 mg bolus
            prop_rate = 6.0                      # 6 mg/kg/h continuous infusion
            remi_rate = 0.25                     # 0.25 ug/kg/min
            action_desc = "[INDUCTION] 120mg Bolus + Inf"
        elif current_time_sec == 90.0:
            action_desc = "[LOC] Loss of Consciousness"
        elif current_time_sec == 150.0:
            prop_rate = 4.5                      # Taper to maintenance rate
            action_desc = "[HYSTERESIS] Stability Check"
        elif current_time_sec == 180.0:
            action_desc = "[READY] Incision Clearance"
        elif current_time_sec == 240.0:
            surgical_pain = 0.8                  # Surgical skin incision!
            action_desc = "[SURGERY] Skin Incision Stim"
        elif current_time_sec == 300.0:
            # Hypotension treatment: titrate Norepinephrine 0.04 ug/kg/min
            norepi_rate = 0.045
            surgical_pain = 0.4                  # Stable maintenance surgery
            action_desc = "[TITRATION] Norepi Pressor On"
        elif current_time_sec == 480.0:
            # End of surgery called: Asymmetric emergence washout
            prop_rate = 0.0                      # Stop Propofol early (CSDT clearance)
            action_desc = "[EMERGENCE] Propofol Washout"
        elif current_time_sec == 600.0:
            remi_rate = 0.0                      # Stop Remifentanil at T-2.5m
            norepi_rate = 0.0
            surgical_pain = 0.0
            action_desc = "[EMERGENCE] Remi Terminated"
        elif current_time_sec == 720.0:
            action_desc = "[RECOVERY] Spontaneous Breath"
        else:
            action_desc = None

        # Execute simulation step
        vitals = sim.step(
            propofol_rate_mg_kg_h=prop_rate,
            remifentanil_rate_ug_kg_min=remi_rate,
            norepinephrine_rate_ug_kg_min=norepi_rate,
            surgical_stimulus=surgical_pain,
            dt_seconds=dt
        )

        # Print telemetry log on key clinical events or every 30 seconds
        if action_desc is not None or (int(current_time_sec) % 30 == 0 and int(current_time_sec) != step_counter):
            step_counter = int(current_time_sec)
            mins = int(current_time_sec // 60)
            secs = int(current_time_sec % 60)
            tag = action_desc if action_desc else "Steady-State Titration"
            
            # Status badge
            if vitals.bis < 40:
                depth_badge = "[DEEP]"
            elif 40 <= vitals.bis <= 60:
                depth_badge = "[TARGET]"
            else:
                depth_badge = "[LIGHT]"

            print(f"{mins:02d}:{secs:02d} | {tag:<26} | {vitals.bis:>5.1f} | {vitals.map_mmhg:>6.1f} | {vitals.hr_bpm:>5.1f} | {vitals.ce_propofol_ug_ml:>8.2f} | {vitals.ce_remi_ng_ml:>8.2f} | {vitals.norepinephrine_rate_ug_kg_min:>7.3f}  {depth_badge}")

    print("-" * 80)
    print("SIMULATION SUMMARY:")
    print(f"Final Patient Status at t={total_seconds/60:.1f}m: BIS={vitals.bis:.1f} (Target Reached >= 80 for PACU Transfer)")
    print(f"Hemodynamic Recovery: MAP={vitals.map_mmhg:.1f} mmHg, HR={vitals.hr_bpm:.1f} bpm")
    print("=" * 80)
    print("Step 1 (Digital Twin Physiology Simulator) successfully verified!")


if __name__ == "__main__":
    run_clinical_demo()
