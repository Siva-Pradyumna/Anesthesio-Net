"""
Stage 1: Deterministic Induction & Readiness Hysteresis Controller
Implements:
1. Target-Controlled Infusion (TCI) induction ramp for rapid Loss of Consciousness (LOC).
2. 30-second continuous physiological dwell lock (Hysteresis Check) before incision clearance.
"""

from dataclasses import dataclass
from typing import Literal
from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.simulator import PatientVitals


@dataclass
class InductionConfig:
    target_ce_prop: float = 3.5          # Target Propofol Ce (ug/mL)
    target_ce_remi: float = 2.5          # Target Remifentanil Ce (ng/mL)
    max_prop_rate: float = 12.0          # Max pump rate during induction (mg/kg/h)
    max_remi_rate: float = 0.5           # Max remi rate during induction (ug/kg/min)
    dwell_duration_sec: float = 30.0     # Required continuous stable duration
    bis_target_min: float = 40.0         # Lower BIS target
    bis_target_max: float = 60.0         # Upper BIS target
    map_target_min: float = 65.0         # Lower MAP target (mmHg)
    map_target_max: float = 100.0        # Upper MAP target (mmHg)


class InductionController:
    """
    Manages Stage 1 of the surgical lifecycle:
    STANDBY -> TCI_RAMPING -> LOC_CONFIRMED -> HYSTERESIS_DWELL -> CLEARANCE_APPROVED
    """

    def __init__(self, patient: PatientProfile, config: InductionConfig | None = None):
        self.patient = patient
        self.config = config or InductionConfig()
        
        self.state: Literal[
            "STANDBY",
            "TCI_RAMPING",
            "LOC_CONFIRMED",
            "HYSTERESIS_DWELL",
            "CLEARANCE_APPROVED"
        ] = "STANDBY"

        self.consecutive_dwell_sec = 0.0
        self.initial_bolus_given = False

    def start_induction(self) -> float:
        """
        Calculates and returns the initial induction bolus dose (mg) of Propofol.
        Transitions state to TCI_RAMPING.
        """
        # Standard clinical induction bolus: ~1.5 to 1.8 mg/kg
        bolus_mg = 1.7 * self.patient.weight_kg
        self.initial_bolus_given = True
        self.state = "TCI_RAMPING"
        self.consecutive_dwell_sec = 0.0
        return bolus_mg

    def compute_infusion_rates(
        self,
        vitals: PatientVitals,
        dt_seconds: float = 1.0
    ) -> tuple[float, float, float, bool]:
        """
        Calculates pump infusion rates and updates the hysteresis dwell lock.
        Returns:
            (propofol_mg_kg_h, remi_ug_kg_min, norepi_ug_kg_min, is_cleared_for_incision)
        """
        cfg = self.config

        if self.state == "STANDBY":
            return 0.0, 0.0, 0.0, False

        # 1. State Progression
        if vitals.bis < 60.0 and self.state == "TCI_RAMPING":
            self.state = "LOC_CONFIRMED"

        # 2. Check if vitals are within target surgical range
        in_bis_bounds = (cfg.bis_target_min <= vitals.bis <= cfg.bis_target_max)
        in_map_bounds = (cfg.map_target_min <= vitals.map_mmhg <= cfg.map_target_max)
        vitals_stable = in_bis_bounds and in_map_bounds

        if self.state in ["LOC_CONFIRMED", "HYSTERESIS_DWELL"]:
            self.state = "HYSTERESIS_DWELL"
            if vitals_stable:
                self.consecutive_dwell_sec += dt_seconds
                if self.consecutive_dwell_sec >= cfg.dwell_duration_sec:
                    self.state = "CLEARANCE_APPROVED"
            else:
                # Chattering prevention: reset dwell timer if vitals breach bounds!
                self.consecutive_dwell_sec = 0.0

        # 3. Closed-Loop TCI Rate Calculations
        # Proportional-Integral feedback toward effect-site targets
        prop_error = cfg.target_ce_prop - vitals.ce_propofol_ug_ml
        remi_error = cfg.target_ce_remi - vitals.ce_remi_ng_ml

        # Maintenance basal rate + error adjustment
        if self.state in ["TCI_RAMPING", "LOC_CONFIRMED"]:
            prop_rate = 6.0 + 3.0 * prop_error
            remi_rate = 0.2 + 0.15 * remi_error
        else:
            # Steady-state dwell maintenance
            prop_rate = 4.5 + 2.0 * prop_error
            remi_rate = 0.15 + 0.1 * remi_error

        # Hemodynamic guardrail: if MAP approaches 65, add micro-pressor
        norepi_rate = 0.0
        if vitals.map_mmhg < 68.0:
            norepi_rate = 0.03 + 0.01 * (68.0 - vitals.map_mmhg)

        # Enforce rate caps
        prop_rate = float(min(max(prop_rate, 0.0), cfg.max_prop_rate))
        remi_rate = float(min(max(remi_rate, 0.0), cfg.max_remi_rate))
        norepi_rate = float(min(max(norepi_rate, 0.0), 0.2))

        is_cleared = (self.state == "CLEARANCE_APPROVED")
        return prop_rate, remi_rate, norepi_rate, is_cleared

    def reset(self) -> None:
        self.state = "STANDBY"
        self.consecutive_dwell_sec = 0.0
        self.initial_bolus_given = False
