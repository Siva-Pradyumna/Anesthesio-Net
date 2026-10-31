"""
Dual-Engine Selector: Mode A (Digital Twin) vs Mode B (VitalDB Replay)
Provides a unified interface to alternate between closed-loop simulation
and real-world clinical dataset streaming in counterfactual shadow mode.
"""

from enum import Enum
from typing import Any
import numpy as np

from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.simulator import PatientSimulator
from anesthesionet.signal.artifact_filter import SignalArtifactFilter, FilteredTelemetry
from anesthesionet.replay.vital_case import generate_benchmark_vitaldb_case
from anesthesionet.replay.replay_engine import VitalDBReplayEngine
from anesthesionet.replay.counterfactual import CounterfactualTransferEstimator, CounterfactualProjection


class ExecutionMode(str, Enum):
    MODE_A_SIMULATOR = "MODE_A_SIMULATOR"
    MODE_B_REPLAY = "MODE_B_REPLAY"


class DualEngineCoordinator:
    """
    Manages telemetry generation across Mode A (Simulation) and Mode B (Replay)
    with integrated signal integrity filtering and counterfactual shadow estimation.
    """

    def __init__(self, patient: PatientProfile | None = None):
        self.patient = patient or PatientProfile(
            age_years=52.0, weight_kg=68.0, height_cm=172.0, sex="female"
        )
        self.mode = ExecutionMode.MODE_A_SIMULATOR

        # Mode A: Digital Twin
        self.simulator = PatientSimulator(self.patient)

        # Mode B: VitalDB Replay Engine & Counterfactual Estimator
        self.replay_case = generate_benchmark_vitaldb_case()
        self.replay_engine = VitalDBReplayEngine(self.replay_case)
        self.counterfactual_estimator = CounterfactualTransferEstimator(self.patient)

        # Signal Integrity Filter
        self.signal_filter = SignalArtifactFilter()

    def switch_mode(self, mode: ExecutionMode | str) -> None:
        """Toggles between Mode A (Simulator) and Mode B (VitalDB Replay)."""
        if isinstance(mode, str):
            mode = ExecutionMode(mode)
        self.mode = mode
        self.signal_filter.reset()

    def step(
        self,
        ai_action: np.ndarray,
        surgical_stimulus: float = 0.0,
        dt_seconds: float = 1.0
    ) -> tuple[FilteredTelemetry, CounterfactualProjection | None, dict[str, Any]]:
        """
        Advances the selected engine by 1 second.
        Returns:
            (filtered_telemetry, counterfactual_projection_or_none, metadata)
        """
        prop_u = float(ai_action[0])
        remi_u = float(ai_action[1])
        norepi_u = float(ai_action[2])

        cf_projection = None
        metadata = {"mode": self.mode.value}

        # ----------------------------------------------------
        # MODE A: DIGITAL TWIN SIMULATION
        # ----------------------------------------------------
        if self.mode == ExecutionMode.MODE_A_SIMULATOR:
            # Advance 10 substeps (0.1s each)
            vitals = None
            for _ in range(int(dt_seconds / 0.1)):
                vitals = self.simulator.step(
                    propofol_rate_mg_kg_h=prop_u,
                    remifentanil_rate_ug_kg_min=remi_u,
                    norepinephrine_rate_ug_kg_min=norepi_u,
                    surgical_stimulus=surgical_stimulus,
                    dt_seconds=0.1
                )

            # Pass through signal integrity filter (clean in Mode A unless external noise injected)
            filtered = self.signal_filter.process(
                time_sec=vitals.time_seconds,
                raw_bis=vitals.bis,
                raw_map=vitals.map_mmhg,
                raw_hr=vitals.hr_bpm,
                sqi_percent=100.0,
                emg_db=22.0,
                dt_seconds=dt_seconds
            )
            metadata["is_finished"] = False

        # ----------------------------------------------------
        # MODE B: REAL-WORLD VITALDB REPLAY ENGINE
        # ----------------------------------------------------
        else:
            frame, is_finished = self.replay_engine.step()

            # Pass through signal integrity filter (evaluates real SQI, EMG, and velocity spikes)
            filtered = self.signal_filter.process(
                time_sec=frame.time_seconds,
                raw_bis=frame.bis,
                raw_map=frame.map_mmhg,
                raw_hr=frame.hr_bpm,
                sqi_percent=frame.sqi_percent,
                emg_db=frame.emg_db,
                dt_seconds=dt_seconds
            )

            # Calculate counterfactual shadow projection (AI recommendation vs historical doctor)
            doctor_action = np.array([
                frame.propofol_rate_doctor,
                frame.remi_rate_doctor,
                frame.norepi_rate_doctor
            ], dtype=np.float32)

            cf_projection = self.counterfactual_estimator.evaluate_step(
                time_sec=frame.time_seconds,
                dataset_bis=filtered.filtered_bis,
                dataset_map=filtered.filtered_map,
                doctor_action=doctor_action,
                ai_action=ai_action,
                dt_seconds=dt_seconds
            )

            metadata["is_finished"] = is_finished
            metadata["event_marker"] = frame.event_marker
            metadata["replay_progress"] = self.replay_engine.progress_percent

        return filtered, cf_projection, metadata

    def reset(self) -> None:
        self.simulator.reset()
        self.replay_engine.reset()
        self.counterfactual_estimator.reset()
        self.signal_filter.reset()
