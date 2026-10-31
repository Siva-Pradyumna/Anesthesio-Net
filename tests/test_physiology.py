"""
Unit Tests for AnesthesioNet-AI Physiology Engine
Verifies mathematical bounds, absence of NaNs, and stability of PK/PD simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anesthesionet.physiology import (
    PatientProfile,
    PatientSimulator,
    compute_eleveld_propofol,
    compute_eleveld_remifentanil,
    PharmacodynamicSynergy
)


def test_patient_covariates():
    p = PatientProfile(age_years=40, weight_kg=70, height_cm=175, sex="male")
    assert round(p.bmi, 1) == 22.9
    assert p.fat_free_mass_janmahasatian > 45.0


def test_eleveld_pk_parameters():
    p = PatientProfile(age_years=35, weight_kg=70, height_cm=170, sex="female")
    prop_pk = compute_eleveld_propofol(p)
    remi_pk = compute_eleveld_remifentanil(p)

    assert prop_pk.V1 > 0
    assert prop_pk.k10 > 0
    assert remi_pk.ke0 > 0


def test_synergy_surface_bounds():
    synergy = PharmacodynamicSynergy()
    
    # Awake state
    bis_awake = synergy.compute_bis(0.0, 0.0)
    assert bis_awake >= 95.0

    # Deep anesthesia
    bis_deep = synergy.compute_bis(5.0, 5.0)
    assert bis_deep < 30.0

    # Target surgical zone (Propofol 2.0 ug/mL + Remifentanil 1.5 ng/mL yields BIS ~ 49)
    bis_target = synergy.compute_bis(2.0, 1.5)
    assert 35.0 <= bis_target <= 65.0


def test_simulator_stability():
    p = PatientProfile(age_years=50, weight_kg=80, height_cm=180, sex="male")
    sim = PatientSimulator(p)

    # 1000 steps of rapid titration
    for _ in range(1000):
        vitals = sim.step(
            propofol_rate_mg_kg_h=5.0,
            remifentanil_rate_ug_kg_min=0.2,
            norepinephrine_rate_ug_kg_min=0.05,
            dt_seconds=0.1
        )
        assert 0.0 <= vitals.bis <= 100.0
        assert 20.0 <= vitals.map_mmhg <= 180.0
        assert 30.0 <= vitals.hr_bpm <= 180.0


if __name__ == "__main__":
    test_patient_covariates()
    test_eleveld_pk_parameters()
    test_synergy_surface_bounds()
    test_simulator_stability()
    print("All physiology unit tests passed successfully!")
