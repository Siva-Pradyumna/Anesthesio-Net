"""
Unit Tests for Step 4: Conformal Predictive Forecasting, Preload Gate, Watchdog & Server
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from anesthesionet.physiology import PatientProfile
from anesthesionet.forecasting import ConformalPredictiveForecaster
from anesthesionet.safety import QualifiedPreloadGate, BedsideWatchdogDaemon
from anesthesionet.server import SurgicalSuiteInstance


def test_conformal_forecaster():
    forecaster = ConformalPredictiveForecaster(coverage_confidence=0.95)
    for s in range(60):
        forecaster.record_frame(
            time_sec=float(s), bis=50.0, map_mmhg=75.0, hr_bpm=70.0,
            prop_rate=3.5, remi_rate=0.12, norepi_rate=0.0
        )
    assessment = forecaster.forecast()
    assert len(assessment.map_forecasts) == 3
    assert len(assessment.bis_forecasts) == 3
    assert assessment.map_forecasts[0].lower_conformal_bound < assessment.map_forecasts[0].upper_conformal_bound


def test_qualified_preload_gate():
    gate = QualifiedPreloadGate(ppv_threshold_percent=13.0)
    
    # Normovolemic test
    normo = gate.evaluate(ppv_percent=8.5, is_mechanical_ventilation=True, tidal_volume_ml_kg=8.5)
    assert not normo.is_fluid_responsive
    assert normo.fluid_bolus_recommended_ml == 0.0

    # Hypovolemic test
    hypo = gate.evaluate(ppv_percent=15.2, is_mechanical_ventilation=True, tidal_volume_ml_kg=8.5)
    assert hypo.is_fluid_responsive
    assert hypo.max_allowable_norepi_rate <= 0.05
    assert hypo.fluid_bolus_recommended_ml == 250.0


def test_bedside_watchdog():
    patient = PatientProfile(age_years=50, weight_kg=70, height_cm=175, sex="male")
    watchdog = BedsideWatchdogDaemon(patient, timeout_threshold_seconds=5.0)

    # Normal heartbeat
    watchdog.receive_heartbeat(current_time=10.0)
    stat = watchdog.evaluate_status(current_time=11.0)
    assert not stat.is_failsafe_tripped

    # Timeout
    stat_timeout = watchdog.evaluate_status(current_time=16.5)
    assert stat_timeout.is_failsafe_tripped
    assert stat_timeout.fallback_propofol_rate_mg_kg_h > 0.0


def test_surgical_suite_instance():
    patient = PatientProfile(age_years=40, weight_kg=70, height_cm=175, sex="male")
    suite = SurgicalSuiteInstance("TEST-OR", patient)
    payload = suite.step()
    assert payload["suite_id"] == "TEST-OR"
    assert "vitals" in payload
    assert "predictions" in payload
    assert "infusion_rates" in payload


if __name__ == "__main__":
    test_conformal_forecaster()
    test_qualified_preload_gate()
    test_bedside_watchdog()
    test_surgical_suite_instance()
    print("All Step 4 unit tests passed successfully!")
