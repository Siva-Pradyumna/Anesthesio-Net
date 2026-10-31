"""
Bedside Hardware Watchdog Daemon
Simulates the local fail-safe safety hardware monitor situated at the patient bedside.
Trips a safety relay if central network communication drops for >= 5.0 seconds,
automatically reverting physical syringe pumps to patient-specific basal TCI maintenance.
"""

from dataclasses import dataclass
import time
from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.eleveld import compute_eleveld_propofol, compute_eleveld_remifentanil


@dataclass
class WatchdogStatus:
    is_connected: bool
    seconds_since_last_heartbeat: float
    is_failsafe_tripped: bool
    fallback_propofol_rate_mg_kg_h: float
    fallback_remi_rate_ug_kg_min: float
    alarm_status: str


class BedsideWatchdogDaemon:
    """
    Independent local bedside hardware monitor supervising cloud/server closed-loop connections.
    """

    def __init__(
        self,
        patient: PatientProfile,
        timeout_threshold_seconds: float = 5.0
    ):
        self.patient = patient
        self.timeout_threshold = timeout_threshold_seconds
        
        # Calculate patient-specific deterministic steady-state basal clearance rates
        # Formula: Infusion = Clearance * Target_Concentration
        prop_pk = compute_eleveld_propofol(patient)
        remi_pk = compute_eleveld_remifentanil(patient)

        # Steady-state basal maintenance targets: Propofol 2.2 ug/mL, Remi 1.5 ng/mL
        # Propofol: Cl1 (L/min) * 2.2 mg/L * 60 min / weight_kg -> mg/kg/h
        self.basal_prop_rate = float((prop_pk.Cl1 * 2.2 * 60.0) / patient.weight_kg)
        self.basal_prop_rate = min(max(self.basal_prop_rate, 2.5), 5.5)

        # Remifentanil: Cl1 (L/min) * 1.5 ug/L / weight_kg -> ug/kg/min
        self.basal_remi_rate = float((remi_pk.Cl1 * 1.5) / patient.weight_kg)
        self.basal_remi_rate = min(max(self.basal_remi_rate, 0.08), 0.25)

        self.last_heartbeat_time = time.time()
        self.is_failsafe_tripped = False

    def receive_heartbeat(self, current_time: float | None = None) -> None:
        """Called upon arrival of an authenticated heartbeat packet from the central server."""
        self.last_heartbeat_time = current_time if current_time is not None else time.time()
        self.is_failsafe_tripped = False

    def evaluate_status(self, current_time: float | None = None) -> WatchdogStatus:
        """
        Evaluates connectivity and trips safety relay upon communication timeout.
        """
        now = current_time if current_time is not None else time.time()
        delta = max(0.0, now - self.last_heartbeat_time)

        if delta >= self.timeout_threshold:
            self.is_failsafe_tripped = True
            alarm = (
                f"[CRITICAL FAILSAFE: HEARTBEAT LOST FOR {delta:.1f}s >= 5.0s — "
                f"NETWORK DECOUPLED — PUMPS LOCKED TO LOCAL BASAL TCI "
                f"(Propofol: {self.basal_prop_rate:.1f} mg/kg/h, Remi: {self.basal_remi_rate:.2f} ug/kg/min)]"
            )
        else:
            alarm = f"[WATCHDOG NORMAL: HEARTBEAT ACTIVE (Latency: {delta*1000:.0f} ms)]"

        return WatchdogStatus(
            is_connected=not self.is_failsafe_tripped,
            seconds_since_last_heartbeat=delta,
            is_failsafe_tripped=self.is_failsafe_tripped,
            fallback_propofol_rate_mg_kg_h=self.basal_prop_rate,
            fallback_remi_rate_ug_kg_min=self.basal_remi_rate,
            alarm_status=alarm
        )

    def reset(self) -> None:
        self.last_heartbeat_time = time.time()
        self.is_failsafe_tripped = False
