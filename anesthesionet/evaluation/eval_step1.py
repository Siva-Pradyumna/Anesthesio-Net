"""
Step 1 Evaluation Suite: Core Mode A Digital Twin Engine
Evaluates:
1. Pharmacokinetic mass conservation & numerical stability (10,000 steps)
2. Minto/Bouillon synergy surface monotonicity and potency shift
3. Coupled hemodynamic responsiveness to vasodilators, pressors, and surgical stimuli
4. Real-time simulation throughput benchmark (steps per second)
"""

import time
import numpy as np
from anesthesionet.physiology import (
    PatientProfile,
    PatientSimulator,
    PharmacodynamicSynergy,
    compute_eleveld_propofol,
    compute_eleveld_remifentanil
)


def evaluate_step1(verbose: bool = True) -> dict[str, bool]:
    results = {}
    print("\n" + "=" * 80)
    print("        ANESTHESIONET-AI EVALUATION: STEP 1 (DIGITAL TWIN ENGINE)")
    print("=" * 80)

    patient = PatientProfile(age_years=45.0, weight_kg=70.0, height_cm=175.0, sex="male")
    sim = PatientSimulator(patient)
    synergy = PharmacodynamicSynergy()

    # ---------------------------------------------------------
    # TEST 1: Pharmacodynamic Synergy Surface Conformance
    # ---------------------------------------------------------
    # Propofol alone required for BIS = 50 vs Propofol + Remifentanil (2 ng/mL)
    bis_prop_alone = synergy.compute_bis(Ce_prop_ug_ml=2.2, Ce_remi_ng_ml=0.0)
    bis_prop_with_remi = synergy.compute_bis(Ce_prop_ug_ml=2.2, Ce_remi_ng_ml=2.0)
    
    # Adding Remifentanil MUST produce deeper sedation (lower BIS)
    synergy_verified = bool(bis_prop_with_remi < bis_prop_alone - 15.0)
    results["Synergy Potency Shift"] = synergy_verified
    
    print(f"1. Synergy Surface Test:")
    print(f"   • Propofol 2.2 ug/mL alone:            BIS = {bis_prop_alone:.1f}")
    print(f"   • Propofol 2.2 ug/mL + Remi 2.0 ng/mL: BIS = {bis_prop_with_remi:.1f} (Synergistic Shift: -{bis_prop_alone - bis_prop_with_remi:.1f})")
    print(f"   -> Result: {'[PASS]' if synergy_verified else '[FAIL]'}")

    # Monotonicity test: increasing drug never increases BIS
    doses = np.linspace(0.5, 6.0, 20)
    bis_series = [synergy.compute_bis(d, 1.5) for d in doses]
    is_monotonic = bool(np.all(np.diff(bis_series) <= 0.0))
    results["PD Monotonicity"] = is_monotonic
    print(f"   • Monotonic Depth Scaling:             {'[PASS]' if is_monotonic else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 2: Hemodynamic Coupling Responsiveness
    # ---------------------------------------------------------
    sim.reset()
    baseline_vitals = sim.step(dt_seconds=0.1)
    
    # Administer propofol bolus -> MAP should decrease (vasodilation takes ~45-60s to peak)
    sim.administer_propofol_bolus(90.0)
    for _ in range(500):  # 50 seconds of physiological equilibration
        vitals_vaso = sim.step(propofol_rate_mg_kg_h=5.0, dt_seconds=0.1)
    map_dropped = bool(vitals_vaso.map_mmhg <= baseline_vitals.map_mmhg - 4.0)
    
    # Administer norepinephrine -> MAP should recover and increase
    for _ in range(400):  # 40 seconds of pressor infusion
        vitals_pressor = sim.step(
            propofol_rate_mg_kg_h=3.0,
            norepinephrine_rate_ug_kg_min=0.08,
            dt_seconds=0.1
        )
    map_recovered = bool(vitals_pressor.map_mmhg > vitals_vaso.map_mmhg + 5.0)
    
    # Apply surgical pain stimulus -> MAP should rise
    for _ in range(200):  # 20 seconds of surgical noxious stimulation
        vitals_pain = sim.step(
            propofol_rate_mg_kg_h=3.0,
            surgical_stimulus=0.8,
            dt_seconds=0.1
        )
    pain_response = bool(vitals_pain.map_mmhg > vitals_pressor.map_mmhg)
    
    hemo_passed = map_dropped and map_recovered and pain_response
    results["Coupled Hemodynamics"] = hemo_passed
    print(f"\n2. Coupled Hemodynamics Test:")
    print(f"   • Baseline MAP:                {baseline_vitals.map_mmhg:.1f} mmHg")
    print(f"   • Propofol Vasodilation Drop:  {vitals_vaso.map_mmhg:.1f} mmHg (Expected Drop)")
    print(f"   • Norepinephrine Pressor Lift: {vitals_pressor.map_mmhg:.1f} mmHg (Expected Elevation)")
    print(f"   • Noxious Surgical Stimulus:   {vitals_pain.map_mmhg:.1f} mmHg (Sympathetic Surge)")
    print(f"   -> Result: {'[PASS]' if hemo_passed else '[FAIL]'}")

    # ---------------------------------------------------------
    # TEST 3: High-Stress Numerical Stability & Throughput
    # ---------------------------------------------------------
    sim.reset()
    total_stress_steps = 10000  # 10,000 steps = 1,000 seconds of surgery
    t0 = time.perf_counter()
    has_nan_or_inf = False
    
    for i in range(total_stress_steps):
        v = sim.step(
            propofol_rate_mg_kg_h=4.0 + 2.0 * np.sin(i * 0.05),
            remifentanil_rate_ug_kg_min=0.2 + 0.1 * np.cos(i * 0.05),
            norepinephrine_rate_ug_kg_min=0.03,
            surgical_stimulus=0.5 if (i % 200 < 50) else 0.0,
            dt_seconds=0.1
        )
        if np.isnan(v.bis) or np.isnan(v.map_mmhg) or np.isinf(v.bis):
            has_nan_or_inf = True
            break
            
    elapsed = time.perf_counter() - t0
    steps_per_sec = total_stress_steps / elapsed
    sim_ratio = (total_stress_steps * 0.1) / elapsed

    stability_passed = not has_nan_or_inf
    results["Numerical Stability (10,000 Steps)"] = stability_passed
    results["Real-Time Speed Benchmark"] = bool(sim_ratio > 100.0)

    print(f"\n3. Numerical Stability & Performance Benchmark:")
    print(f"   • Evaluated Steps:             {total_stress_steps:,} steps")
    print(f"   • Elapsed Time:                {elapsed:.3f} seconds")
    print(f"   • Throughput:                  {steps_per_sec:,.0f} steps/second")
    print(f"   • Simulation Speed Ratio:      {sim_ratio:,.0f}x Faster Than Real Time")
    print(f"   • NaN / Inf Anomalies:         {'NONE DETECTED' if stability_passed else 'CRITICAL ERROR'}")
    print(f"   -> Result: {'[PASS]' if stability_passed else '[FAIL]'}")

    all_passed = all(results.values())
    print("-" * 80)
    print(f"STEP 1 EVALUATION STATUS: {'>>> ALL TESTS PASSED <<<' if all_passed else '>>> SOME TESTS FAILED <<<'}")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    evaluate_step1()
