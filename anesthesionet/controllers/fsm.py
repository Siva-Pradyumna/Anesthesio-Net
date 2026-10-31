"""
Full-Lifecycle Finite State Machine (FSM) Controller
Coordinates the clinical anesthesia lifecycle:
STANDBY -> STAGE 1 (Induction) -> READINESS HYSTERESIS -> STAGE 2 (Maintenance) -> STAGE 3 (Emergence) -> PACU TRANSFER
"""

from enum import Enum
from typing import Callable, Any
import numpy as np

from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.simulator import PatientVitals
from anesthesionet.controllers.induction import InductionController
from anesthesionet.safety.hocbf_filter import HOCBFSafetyFilter


class SurgicalStage(str, Enum):
    STANDBY = "STANDBY"
    STAGE_1_INDUCTION = "STAGE_1_INDUCTION"
    READINESS_HYSTERESIS = "READINESS_HYSTERESIS"
    STAGE_2_MAINTENANCE = "STAGE_2_MAINTENANCE"
    STAGE_3_EMERGENCE = "STAGE_3_EMERGENCE"
    PACU_TRANSFER = "PACU_TRANSFER"


class FullLifecycleController:
    """
    Supervises full lifecycle transitions and controller handoffs.
    """

    def __init__(
        self,
        patient: PatientProfile,
        maintenance_policy: Callable[[PatientVitals], np.ndarray] | None = None
    ):
        self.patient = patient
        self.stage = SurgicalStage.STANDBY
        
        # Sub-controllers
        self.induction_controller = InductionController(patient)
        self.safety_filter = HOCBFSafetyFilter()
        
        # Maintenance policy (default: steady-state closed-loop titration)
        self.maintenance_policy = maintenance_policy or self._default_maintenance_policy

        # Emergence tracking
        self.emergence_start_time = 0.0
        self.target_emergence_duration = 300.0  # 5 minutes countdown

    def trigger_induction(self) -> float:
        """Clinician triggers start of induction. Returns bolus dose in mg."""
        self.stage = SurgicalStage.STAGE_1_INDUCTION
        return self.induction_controller.start_induction()

    def trigger_emergence(self, current_time: float) -> None:
        """Clinician calls 'End of Surgery'."""
        self.stage = SurgicalStage.STAGE_3_EMERGENCE
        self.emergence_start_time = current_time

    def step(
        self,
        vitals: PatientVitals,
        surgical_stimulus: float = 0.0,
        dt_seconds: float = 1.0
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """
        Advances the FSM and returns the certified safe infusion actions:
        (propofol_mg_kg_h, remi_ug_kg_min, norepi_ug_kg_min)
        """
        was_safety_intervened = False
        info = {
            "stage": self.stage.value,
            "dwell_sec": self.induction_controller.consecutive_dwell_sec,
            "badge": None
        }

        # ----------------------------------------------------
        # STAGE 1: INDUCTION & READINESS CHECK
        # ----------------------------------------------------
        if self.stage in [SurgicalStage.STAGE_1_INDUCTION, SurgicalStage.READINESS_HYSTERESIS]:
            prop, remi, norepi, is_cleared = self.induction_controller.compute_infusion_rates(
                vitals, dt_seconds=dt_seconds
            )
            raw_action = np.array([prop, remi, norepi], dtype=np.float32)

            if self.induction_controller.state == "HYSTERESIS_DWELL":
                self.stage = SurgicalStage.READINESS_HYSTERESIS
                info["badge"] = f"[HYSTERESIS LOCK: {self.induction_controller.consecutive_dwell_sec:.0f}s / 30s]"

            if is_cleared:
                # Transition to Stage 2 Maintenance
                self.stage = SurgicalStage.STAGE_2_MAINTENANCE
                info["badge"] = "[INDUCTION COMPLETE — SAFE FOR SURGICAL INCISION]"

            # Run safety filter
            safe_action, was_safety_intervened, _ = self.safety_filter.filter_action(
                raw_action, vitals, surgical_stimulus=surgical_stimulus
            )
            return safe_action, was_safety_intervened, info

        # ----------------------------------------------------
        # STAGE 2: MAINTENANCE
        # ----------------------------------------------------
        elif self.stage == SurgicalStage.STAGE_2_MAINTENANCE:
            raw_action = self.maintenance_policy(vitals)
            safe_action, was_safety_intervened, _ = self.safety_filter.filter_action(
                raw_action, vitals, surgical_stimulus=surgical_stimulus
            )
            info["badge"] = "[MAINTENANCE CLOSED-LOOP ACTIVE]"
            return safe_action, was_safety_intervened, info

        # ----------------------------------------------------
        # STAGE 3: EMERGENCE & RECOVERY
        # ----------------------------------------------------
        elif self.stage == SurgicalStage.STAGE_3_EMERGENCE:
            time_in_emergence = vitals.time_seconds - self.emergence_start_time
            countdown_remaining = max(0.0, self.target_emergence_duration - time_in_emergence)

            # Asymmetric Washout:
            # Propofol cut to 0.0 immediately
            prop_rate = 0.0
            norepi_rate = 0.0

            # Remifentanil maintained low for ETT tolerance until T - 3 min (180s remaining)
            if countdown_remaining > 180.0:
                remi_rate = 0.05  # Low basal infusion for airway tolerance
                info["badge"] = f"[EMERGENCE: ETT TOLERANCE (Remi 0.05) | Countdown: {countdown_remaining:.0f}s]"
            else:
                remi_rate = 0.0   # Discontinued at T-3m for spontaneous respiration drive
                info["badge"] = f"[EMERGENCE: RESPIRATION CLEARANCE | Countdown: {countdown_remaining:.0f}s]"

            if vitals.bis >= 80.0 and countdown_remaining <= 60.0:
                self.stage = SurgicalStage.PACU_TRANSFER
                info["badge"] = "[PATIENT CONSCIOUS & STABLE — READY FOR PACU TRANSFER]"

            return np.array([prop_rate, remi_rate, norepi_rate], dtype=np.float32), False, info

        # ----------------------------------------------------
        # PACU TRANSFER / STANDBY
        # ----------------------------------------------------
        else:
            return np.zeros(3, dtype=np.float32), False, info

    def _default_maintenance_policy(self, vitals: PatientVitals) -> np.ndarray:
        """
        Proportional closed-loop maintenance policy targeting BIS = 50 and MAP = 75.
        """
        bis_error = vitals.bis - 50.0
        map_error = 75.0 - vitals.map_mmhg

        # Propofol adjustment
        prop_rate = 4.0 + 0.15 * bis_error
        # Remifentanil adjustment
        remi_rate = 0.20 + 0.005 * bis_error
        # Norepinephrine adjustment
        norepi_rate = 0.0
        if map_error > 5.0:
            norepi_rate = 0.02 + 0.005 * map_error

        return np.array([
            np.clip(prop_rate, 1.0, 8.0),
            np.clip(remi_rate, 0.05, 0.5),
            np.clip(norepi_rate, 0.0, 0.15)
        ], dtype=np.float32)

    def reset(self) -> None:
        self.stage = SurgicalStage.STANDBY
        self.induction_controller.reset()
        self.emergence_start_time = 0.0
