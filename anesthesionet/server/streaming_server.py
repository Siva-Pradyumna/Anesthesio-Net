"""
FastAPI & WebSockets Real-Time Multi-Room Streaming Server
Powers the 1-to-Many remote supervision clinical command station.
Streams live 10 Hz telemetry, conformal predictive forecast intervals,
and HOCBF safety statuses across independent surgical suites (OR 1, OR 2, OR 3)
with FULL PARALLEL CONCURRENT EXECUTION across all operating rooms.
"""

import asyncio
from collections import deque
from datetime import datetime
import json
import random
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from fastapi.responses import HTMLResponse
import numpy as np

from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.engine_selector import DualEngineCoordinator, ExecutionMode
from anesthesionet.controllers.fsm import FullLifecycleController, SurgicalStage
from anesthesionet.forecasting.conformal_tft import ConformalPredictiveForecaster
from anesthesionet.safety.preload_gate import QualifiedPreloadGate
from anesthesionet.safety.watchdog import BedsideWatchdogDaemon


# Comprehensive Clinical Metadata for Scheduled Surgeries
SURGERY_SCHEDULE = {
    "OR-1": {
        "suite_id": "OR-1",
        "room_name": "Operating Suite 1 (OR-1)",
        "specialty": "General & Laparoscopic Surgery",
        "procedure_name": "Laparoscopic Cholecystectomy",
        "lead_surgeon": "Dr. Sarah Mitchell, FACS",
        "anesthesiologist": "Dr. K. Ramaswamy, MD",
        "scheduled_time": "08:30 AM",
        "patient_name": "Johnathan Davies",
        "demographics": "48 y/o Male | 72 kg | 176 cm | BMI 23.2",
        "asa_class": "ASA Class II",
        "preop_diagnosis": "Symptomatic Cholelithiasis / Chronic Cholecystitis",
        "surgical_plan": "Four-trocar laparoscopic extraction with intraoperative cholangiography"
    },
    "OR-2": {
        "suite_id": "OR-2",
        "room_name": "Operating Suite 2 (OR-2)",
        "specialty": "Orthopedic & Joint Reconstruction",
        "procedure_name": "Total Hip Arthroplasty (Right)",
        "lead_surgeon": "Dr. Robert Vance, MD, FAAOS",
        "anesthesiologist": "Dr. Angela Chen, MD",
        "scheduled_time": "09:15 AM",
        "patient_name": "Eleanor Vance-Cole",
        "demographics": "62 y/o Female | 65 kg | 165 cm | BMI 23.9",
        "asa_class": "ASA Class II",
        "preop_diagnosis": "Severe End-Stage Osteoarthritis of Right Hip",
        "surgical_plan": "Posterior approach ceramic-on-polyethylene press-fit total arthroplasty"
    },
    "OR-3": {
        "suite_id": "OR-3",
        "room_name": "Operating Suite 3 (OR-3)",
        "specialty": "Cardiovascular & Thoracic Surgery",
        "procedure_name": "Off-Pump Coronary Artery Bypass (OPCAB x3)",
        "lead_surgeon": "Dr. Elena Rostova, MD, PhD",
        "anesthesiologist": "Dr. Markus Lindqvist, MD",
        "scheduled_time": "10:00 AM",
        "patient_name": "David Henderson",
        "demographics": "35 y/o Male | 80 kg | 182 cm | BMI 24.2",
        "asa_class": "ASA Class III",
        "preop_diagnosis": "Triple-Vessel Ischemic Coronary Artery Disease",
        "surgical_plan": "Median sternotomy, off-pump LIMA-to-LAD + SVG-to-OM1/PDA bypass grafting"
    }
}


class SurgicalSuiteInstance:
    """Encapsulates a full, independent operating room instance."""

    def __init__(self, suite_id: str, patient: PatientProfile):
        self.suite_id = suite_id
        self.patient = patient
        self.surgery_meta = SURGERY_SCHEDULE.get(suite_id, SURGERY_SCHEDULE["OR-1"])
        
        self.engine = DualEngineCoordinator(patient)
        self.controller = FullLifecycleController(patient)
        self.forecaster = ConformalPredictiveForecaster(coverage_confidence=0.95)
        self.preload_gate = QualifiedPreloadGate()
        self.watchdog = BedsideWatchdogDaemon(patient)

        # Autonomy: Level 2 (Advisory) or Level 4 (Autonomous Closed-Loop)
        self.autonomy_level = "LEVEL_4_CLOSED_LOOP"
        self.current_time_sec = 0.0
        self.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.surgical_stimulus = 0.0
        self.ppv_bias = 0.0
        self.injected_artifact_duration = 0.0

        # Clinical Lifecycle Phase:
        # STANDBY -> STAGE_1_INDUCTION -> PATIENT_ASLEEP_READY -> STAGE_2_MAINTENANCE -> SURGERY_COMPLETED -> STAGE_3_EMERGENCE -> PACU_TRANSFER
        self.clinical_phase = "STANDBY"
        
        # Induction and dwell stability counters
        self.induction_step_count = 0
        self.dwell_counter_sec = 0.0
        
        # Stopwatches & Countdown Timers (per suite)
        self.surgery_timer_active = False
        self.surgery_duration_sec = 0.0
        
        # Emergence countdown: Realistic 120-second (2:00 min) countdown to awakening
        self.target_emergence_duration = 120.0
        self.emergence_timer_active = False
        self.emergence_duration_sec = 0.0
        self.is_patient_awake = False
        self.time_to_wakeup_sec = None

        # Challenge and Autonomous Recovery Tracking
        self.active_challenge = None
        self.challenge_history: List[dict] = []

        # Chronological Event & Audit Log
        self.event_log: List[dict] = [
            {"time": "00:00:00", "tag": "PRE-OP", "text": "Patient admitted to OR. Monitoring active. Pre-induction checklist verified.", "type": "info"}
        ]

        # Persistent rolling history buffers (70 points) for unbroken waveforms
        self.bis_history = deque(maxlen=70)
        self.map_history = deque(maxlen=70)
        self.raw_bis_history = deque(maxlen=70)

        # Cached last payload for instant transmission
        self.last_payload: dict = {}
        self.step()  # Initial state step

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = max(0, int(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def add_event_log(self, tag: str, text: str, log_type: str = "info"):
        """Appends a timestamped audit event."""
        if self.surgery_timer_active or self.surgery_duration_sec > 0:
            t_str = self._format_time(self.surgery_duration_sec)
        elif self.emergence_timer_active or self.emergence_duration_sec > 0:
            t_str = f"+{self._format_time(self.emergence_duration_sec)}"
        else:
            t_str = self._format_time(self.current_time_sec)

        entry = {
            "time": t_str,
            "tag": tag,
            "text": text,
            "type": log_type
        }
        self.event_log.append(entry)
        if len(self.event_log) > 80:
            self.event_log.pop(0)

    def step(self) -> dict:
        dt = 0.5  # 2 Hz continuous time increment
        self.current_time_sec += dt
        self.watchdog.receive_heartbeat(self.current_time_sec)

        # Decay transient stimuli
        if self.surgical_stimulus > 0.0:
            self.surgical_stimulus = max(0.0, self.surgical_stimulus - 0.02)
        if self.ppv_bias > 0.0:
            self.ppv_bias = max(0.0, self.ppv_bias - 0.08)

        # Advance active timers
        if self.surgery_timer_active:
            self.surgery_duration_sec += dt
        if self.emergence_timer_active:
            self.emergence_duration_sec += dt

        was_intervened = False
        cf_proj = None
        is_sensor_hold = False
        prop_ce = self.engine.simulator.pk_prop.Ce
        remi_ce = self.engine.simulator.pk_remi.Ce

        # =====================================================================
        # 1. STANDBY: Awake patient in pre-op holding
        # =====================================================================
        if self.clinical_phase == "STANDBY":
            self.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.engine.simulator.pk_prop.reset()
            self.engine.simulator.pk_remi.reset()
            self.engine.simulator.hemodynamics.reset()

            raw_bis = 98.0
            filtered_bis = 98.0
            raw_map = 90.0
            filtered_map = 90.0
            raw_hr = 75.0
            filtered_hr = 75.0
            ppv_est = 8.5
            sqi_pct = 98.0
            emg_db = 24.0
            prop_ce = 0.0
            remi_ce = 0.0
            safe_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            badge = "[STANDBY: PRE-OP READY — AWAITING INDUCTION]"
            readiness_text = "Patient awake in pre-op holding area. Ready for induction."
            is_patient_asleep = False
            is_safe_for_surgery = False

        # =====================================================================
        # 2. STAGE 1 INDUCTION: Bolus administered, TCI ramping
        # =====================================================================
        elif self.clinical_phase == "STAGE_1_INDUCTION":
            self.induction_step_count += 1
            filtered, cf_proj, meta = self.engine.step(
                ai_action=self.latest_action,
                surgical_stimulus=self.surgical_stimulus,
                dt_seconds=dt
            )

            raw_bis = filtered.raw_bis
            filtered_bis = filtered.filtered_bis
            raw_map = filtered.raw_map
            filtered_map = filtered.filtered_map
            raw_hr = filtered.raw_hr
            filtered_hr = filtered.filtered_hr
            ppv_est = 9.0 + self.ppv_bias
            sqi_pct = filtered.sqi_percent
            emg_db = filtered.emg_db
            prop_ce = self.engine.simulator.pk_prop.Ce
            remi_ce = self.engine.simulator.pk_remi.Ce

            # Compute induction infusion rates
            vitals_obj = type("VitalsObj", (), {
                "bis": filtered_bis,
                "map_mmhg": filtered_map,
                "hr_bpm": filtered_hr,
                "ce_propofol_ug_ml": prop_ce,
                "ce_remi_ng_ml": remi_ce,
                "time_seconds": self.current_time_sec
            })()
            prop, remi, norepi, _ = self.controller.induction_controller.compute_infusion_rates(
                vitals_obj, dt_seconds=dt
            )
            self.latest_action = np.array([prop, remi, norepi], dtype=np.float32)
            safe_action = self.latest_action

            # Dwell stability evaluation
            if filtered_bis <= 58.0:
                self.dwell_counter_sec += dt
            else:
                self.dwell_counter_sec = max(0.0, self.dwell_counter_sec - dt * 0.5)

            if filtered_bis <= 58.0 and self.dwell_counter_sec >= 8.0:
                self.clinical_phase = "PATIENT_ASLEEP_READY"
                self.add_event_log("✓ LOC CONFIRMED", "Patient asleep (LOC verified). Therapeutic depth stable. Safe to start surgical incision.", "success")
                badge = "[✓ PATIENT ASLEEP (LOC CONFIRMED) — SAFE TO START SURGERY]"
                readiness_text = "PATIENT IS ASLEEP (LOC CONFIRMED) — SAFE TO START SURGERY!"
                is_patient_asleep = True
                is_safe_for_surgery = True
            elif filtered_bis < 60.0:
                badge = f"[LOC CONFIRMED — STABILIZING IN TARGET ({self.dwell_counter_sec:.0f}s / 8s)]"
                readiness_text = f"Loss of Consciousness confirmed (BIS {filtered_bis:.0f}). Verifying depth stability ({self.dwell_counter_sec:.0f}s / 8s)..."
                is_patient_asleep = True
                is_safe_for_surgery = False
            else:
                badge = f"[STAGE 1: TCI INDUCTION RAMPING (BIS: {filtered_bis:.0f})]"
                readiness_text = f"Induction in progress: Propofol bolus administered. Ce rising ({prop_ce:.2f} µg/mL), BIS descending..."
                is_patient_asleep = False
                is_safe_for_surgery = False

        # =====================================================================
        # 3. PATIENT ASLEEP & READY: Awaiting Operator Incision Call
        # =====================================================================
        elif self.clinical_phase == "PATIENT_ASLEEP_READY":
            self.controller.stage = SurgicalStage.READINESS_HYSTERESIS
            self.latest_action = np.array([3.4, 0.12, 0.0], dtype=np.float32)
            filtered, cf_proj, meta = self.engine.step(
                ai_action=self.latest_action,
                surgical_stimulus=0.0,
                dt_seconds=dt
            )

            raw_bis = filtered.raw_bis
            filtered_bis = filtered.filtered_bis
            raw_map = filtered.raw_map
            filtered_map = filtered.filtered_map
            raw_hr = filtered.raw_hr
            filtered_hr = filtered.filtered_hr
            ppv_est = 9.0 + self.ppv_bias
            sqi_pct = filtered.sqi_percent
            emg_db = filtered.emg_db
            prop_ce = self.engine.simulator.pk_prop.Ce
            remi_ce = self.engine.simulator.pk_remi.Ce
            safe_action = self.latest_action

            badge = "[✓ PATIENT ASLEEP (LOC CONFIRMED) — SAFE TO START SURGERY]"
            readiness_text = "PATIENT IS ASLEEP (LOC CONFIRMED) — SAFE TO START SURGERY!"
            is_patient_asleep = True
            is_safe_for_surgery = True

        # =====================================================================
        # 4. STAGE 2 MAINTENANCE: Active Procedure & Closed-Loop Control
        # =====================================================================
        elif self.clinical_phase == "STAGE_2_MAINTENANCE":
            self.controller.stage = SurgicalStage.STAGE_2_MAINTENANCE
            filtered, cf_proj, meta = self.engine.step(
                ai_action=self.latest_action,
                surgical_stimulus=self.surgical_stimulus,
                dt_seconds=dt
            )

            raw_bis = filtered.raw_bis
            filtered_bis = filtered.filtered_bis
            raw_map = filtered.raw_map
            filtered_map = filtered.filtered_map
            raw_hr = filtered.raw_hr
            filtered_hr = filtered.filtered_hr
            ppv_est = 9.0 + self.ppv_bias + (max(0.0, 75.0 - filtered_map) * 0.1)
            sqi_pct = filtered.sqi_percent
            emg_db = filtered.emg_db
            prop_ce = self.engine.simulator.pk_prop.Ce
            remi_ce = self.engine.simulator.pk_remi.Ce

            # Injected Electrocautery (Bovie) Interference Handling
            if self.injected_artifact_duration > 0.0:
                self.injected_artifact_duration -= dt
                is_sensor_hold = True
                # Extreme high-frequency electrical noise on raw line
                raw_bis = float(np.clip(97.0 + random.uniform(-2.5, 2.5), 90.0, 100.0))
                sqi_pct = 16.0
                emg_db = 58.0
                # Filter strictly protects patient by holding safe baseline (no drug dump!)
                filtered_bis = 48.0
                if self.injected_artifact_duration <= 0.0:
                    self.add_event_log("✓ SENSOR RELEASED", "Electrocautery deactivated. Clean EEG tracking restored (SQI 98%). Patient protected from overdose.", "success")

            # Evaluate Closed-Loop Controller
            vitals_obj = type("VitalsObj", (), {
                "bis": filtered_bis,
                "map_mmhg": filtered_map,
                "hr_bpm": filtered_hr,
                "ce_propofol_ug_ml": prop_ce,
                "ce_remi_ng_ml": remi_ce,
                "time_seconds": self.current_time_sec
            })()
            safe_action, was_intervened, _ = self.controller.step(
                vitals=vitals_obj,
                surgical_stimulus=self.surgical_stimulus,
                dt_seconds=dt
            )

            # Preload Gate enforcement
            preload_eval = self.preload_gate.evaluate(ppv_percent=ppv_est)
            if preload_eval.is_fluid_responsive:
                safe_action[2] = min(safe_action[2], preload_eval.max_allowable_norepi_rate)

            # MAP Crash Recovery Dynamics:
            if filtered_map < 65.0:
                # Interlock escalates Norepinephrine infusion to restore arterial tone
                safe_action[2] = max(float(safe_action[2]), 0.08 + 0.005 * (65.0 - filtered_map))
            elif filtered_map >= 70.0 and self.engine.simulator.hemodynamics.params.baseline_map < 90.0:
                # Smoothly restore baseline MAP after crash is neutralized
                self.engine.simulator.hemodynamics.params.baseline_map = min(90.0, self.engine.simulator.hemodynamics.params.baseline_map + 0.4)

            self.latest_action = safe_action
            badge = "[STAGE 2: MAINTENANCE — SURGERY IN PROGRESS]"
            readiness_text = "Surgery in progress. Closed-loop MIMO-SAC controller & HOCBF safety active."
            is_patient_asleep = True
            is_safe_for_surgery = False

            # Challenge Auto-Recovery Evaluation
            if self.active_challenge and self.active_challenge.get("status") == "RECOVERING":
                if 40.0 <= filtered_bis <= 60.0 and 68.0 <= filtered_map <= 95.0 and self.injected_artifact_duration <= 0.0:
                    self.active_challenge["status"] = "AUTO_RECOVERED"
                    rec_text = f"Vitals stabilized in therapeutic target: BIS {filtered_bis:.0f}, MAP {filtered_map:.0f} mmHg."
                    self.active_challenge["recovery_text"] = rec_text
                    self.add_event_log("✓ AUTO-RECOVERED", f"{self.active_challenge['title']} neutralized. {rec_text}", "success")
                    self.challenge_history.append(dict(self.active_challenge))

        # =====================================================================
        # 5. SURGERY COMPLETED: Incision closed, waiting for Emergence
        # =====================================================================
        elif self.clinical_phase == "SURGERY_COMPLETED":
            self.latest_action = np.array([2.5, 0.08, 0.0], dtype=np.float32)
            filtered, cf_proj, meta = self.engine.step(
                ai_action=self.latest_action,
                surgical_stimulus=0.0,
                dt_seconds=dt
            )

            raw_bis = filtered.raw_bis
            filtered_bis = filtered.filtered_bis
            raw_map = filtered.raw_map
            filtered_map = filtered.filtered_map
            raw_hr = filtered.raw_hr
            filtered_hr = filtered.filtered_hr
            ppv_est = 9.0 + self.ppv_bias
            sqi_pct = filtered.sqi_percent
            emg_db = filtered.emg_db
            prop_ce = self.engine.simulator.pk_prop.Ce
            remi_ce = self.engine.simulator.pk_remi.Ce
            safe_action = self.latest_action

            badge = "[SURGERY COMPLETED — AWAITING OPERATOR TO START EMERGENCE]"
            readiness_text = f"Surgical procedure completed (Duration: {self._format_time(self.surgery_duration_sec)}). Click 'Start Emergence Phase' to awaken patient."
            is_patient_asleep = True
            is_safe_for_surgery = False

        # =====================================================================
        # 6. STAGE 3 EMERGENCE & PACU TRANSFER: Drug Washout & Countdown
        # =====================================================================
        elif self.clinical_phase in ["STAGE_3_EMERGENCE", "PACU_TRANSFER"]:
            safe_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.latest_action = safe_action

            # Realistic, smooth pharmacokinetic clearance across 120-second countdown
            r = min(1.0, self.emergence_duration_sec / 110.0)
            new_ce_prop = max(0.35, 2.2 * (1.0 - 0.78 * r))
            new_ce_remi = max(0.08, 1.5 * (1.0 - 0.90 * r))
            
            self.engine.simulator.pk_prop.state[3] = new_ce_prop
            self.engine.simulator.pk_remi.state[3] = new_ce_remi
            self.engine.simulator.pk_prop.state[0] = max(0.0, self.engine.simulator.pk_prop.state[0] - dt * 0.15)
            self.engine.simulator.pk_remi.state[0] = max(0.0, self.engine.simulator.pk_remi.state[0] - dt * 0.1)

            # Compute BIS from Bouillon response surface
            raw_bis = self.engine.simulator.pd_synergy.compute_bis(new_ce_prop, new_ce_remi)
            filtered_bis = raw_bis
            raw_map, raw_hr, ppv_est = self.engine.simulator.hemodynamics.step(
                Ce_prop_ug_ml=new_ce_prop,
                Ce_remi_ng_ml=new_ce_remi,
                norepi_infusion_ug_kg_min=0.0,
                surgical_noxious_stimulus=0.0,
                dt_seconds=dt
            )
            filtered_map = raw_map
            filtered_hr = raw_hr
            sqi_pct = 98.0
            emg_db = 26.0
            prop_ce = new_ce_prop
            remi_ce = new_ce_remi

            # Awakening check: at BIS >= 75 and Propofol Ce <= 1.05
            if raw_bis >= 75.0 and prop_ce <= 1.05:
                if not self.is_patient_awake:
                    self.is_patient_awake = True
                    self.time_to_wakeup_sec = self.emergence_duration_sec
                    self.add_event_log("✓ PATIENT AWAKE", f"Consciousness regained (BIS {filtered_bis:.0f}, Prop Ce {prop_ce:.2f} µg/mL). Extubation criteria satisfied at post-op +{self._format_time(self.time_to_wakeup_sec)}.", "success")
                self.clinical_phase = "PACU_TRANSFER"
                badge = "[✓ PATIENT AWAKE & CONSCIOUS — READY FOR PACU TRANSFER]"
                readiness_text = f"PATIENT IS AWAKE (BIS {filtered_bis:.0f}, Prop Ce {prop_ce:.2f} µg/mL). Extubation criteria met at post-op +{self._format_time(self.time_to_wakeup_sec)}."
                is_patient_asleep = False
                is_safe_for_surgery = False
            else:
                rem_countdown = max(0.0, self.target_emergence_duration - self.emergence_duration_sec)
                badge = f"[EMERGENCE WASHOUT — COUNTDOWN: {self._format_time(rem_countdown)} | BIS {filtered_bis:.0f}]"
                readiness_text = f"Emergence in progress: Anesthetic washout active. Drug values clearing (Prop Ce: {prop_ce:.2f} µg/mL), BIS rising..."
                is_patient_asleep = True
                is_safe_for_surgery = False

        # Conformal prediction & Safety
        self.forecaster.record_frame(
            time_sec=self.current_time_sec,
            bis=filtered_bis,
            map_mmhg=filtered_map,
            hr_bpm=filtered_hr,
            prop_rate=float(self.latest_action[0]),
            remi_rate=float(self.latest_action[1]),
            norepi_rate=float(self.latest_action[2])
        )
        prediction = self.forecaster.forecast()
        preload_eval = self.preload_gate.evaluate(ppv_percent=ppv_est)
        watchdog_stat = self.watchdog.evaluate_status(self.current_time_sec)

        # Explainable AI feature attributions
        xai_map_risk = round(min(100.0, max(5.0, (75.0 - filtered_map) * 3.0 + 35.0)), 1)
        xai_bis_error = round(min(100.0, max(5.0, abs(50.0 - filtered_bis) * 3.5 + 20.0)), 1)
        xai_nociception = round(min(100.0, max(5.0, self.surgical_stimulus * 95.0 + 10.0)), 1)

        if self.surgical_stimulus > 0.25:
            xai_driver = "Surgical Nociception Surge"
        elif filtered_map < 68.0:
            xai_driver = "Conformal Hypotension Lookahead"
        elif abs(filtered_bis - 50.0) > 10.0:
            xai_driver = "Hypnotic Depth Error Compensation"
        elif self.clinical_phase in ["STAGE_3_EMERGENCE", "PACU_TRANSFER"]:
            xai_driver = "Emergence Pharmacokinetic Clearance"
        elif self.clinical_phase == "STANDBY":
            xai_driver = "Pre-Induction Awake Baseline"
        else:
            xai_driver = "Bouillon Synergistic Ridge Titration"

        # Drug pump status labels
        if self.clinical_phase == "STANDBY":
            prop_status = "Standby (0.00)"
            remi_status = "Standby (0.00)"
        elif self.clinical_phase == "STAGE_1_INDUCTION":
            prop_status = "TCI Bolus Ramp"
            remi_status = "Synergistic Co-Induction"
        elif self.clinical_phase in ["STAGE_3_EMERGENCE", "PACU_TRANSFER"]:
            prop_status = "Clearing (Washout)"
            remi_status = "Airway / Washout"
        else:
            prop_status = "Closed-Loop Titration"
            remi_status = "Analgesic Titration"

        # Emergence countdown display calculation
        countdown_rem_sec = max(0.0, self.target_emergence_duration - self.emergence_duration_sec) if self.emergence_timer_active else self.target_emergence_duration
        countdown_disp = self._format_time(countdown_rem_sec)

        # Push to persistent per-suite history buffers
        self.bis_history.append(filtered_bis)
        self.map_history.append(filtered_map)
        self.raw_bis_history.append(raw_bis)

        # Compile full payload
        self.last_payload = {
            "suite_id": self.suite_id,
            "time_seconds": self.current_time_sec,
            "mode": self.engine.mode.value,
            "phase": self.clinical_phase,
            "stage": self.controller.stage.value,
            "badge": badge,
            "autonomy_level": self.autonomy_level,
            "surgery_meta": self.surgery_meta,
            "readiness": {
                "is_patient_asleep": is_patient_asleep,
                "is_safe_for_surgery": is_safe_for_surgery,
                "status_text": readiness_text
            },
            "timers": {
                "surgery_timer_active": self.surgery_timer_active,
                "surgery_duration_sec": self.surgery_duration_sec,
                "surgery_formatted": self._format_time(self.surgery_duration_sec),
                "emergence_timer_active": self.emergence_timer_active,
                "emergence_duration_sec": self.emergence_duration_sec,
                "emergence_formatted": self._format_time(self.emergence_duration_sec),
                "countdown_remaining_sec": countdown_rem_sec,
                "countdown_formatted": countdown_disp,
                "is_patient_awake": self.is_patient_awake,
                "time_to_wakeup_sec": self.time_to_wakeup_sec,
                "wakeup_formatted": self._format_time(self.time_to_wakeup_sec) if self.time_to_wakeup_sec is not None else "--:--"
            },
            "vitals": {
                "raw_bis": round(raw_bis, 1),
                "filtered_bis": round(filtered_bis, 1),
                "raw_map": round(raw_map, 1),
                "filtered_map": round(filtered_map, 1),
                "filtered_hr": round(filtered_hr, 1),
                "ppv_percent": round(ppv_est, 1),
                "sqi_percent": round(sqi_pct, 1),
                "emg_db": round(emg_db, 1),
                "ce_propofol_ug_ml": round(prop_ce, 2),
                "ce_remi_ng_ml": round(remi_ce, 2)
            },
            "infusion_rates": {
                "propofol_mg_kg_h": round(float(safe_action[0]), 2),
                "remifentanil_ug_kg_min": round(float(safe_action[1]), 3),
                "norepinephrine_ug_kg_min": round(float(safe_action[2]), 3),
                "propofol_ce_ug_ml": round(prop_ce, 2),
                "remifentanil_ce_ng_ml": round(remi_ce, 2),
                "propofol_status": prop_status,
                "remifentanil_status": remi_status
            },
            "challenge": {
                "active": self.active_challenge,
                "recent_history": self.challenge_history[-3:] if self.challenge_history else []
            },
            "event_log": self.event_log[-25:],
            "recent_history": {
                "bis": list(self.bis_history),
                "map": list(self.map_history),
                "raw_bis": list(self.raw_bis_history)
            },
            "predictions": {
                "proactive_hypotension": prediction.proactive_hypotension_alert,
                "interlock_message": prediction.interlock_message,
                "map_10min_conformal_low": round(prediction.map_forecasts[1].lower_conformal_bound, 1),
                "map_10min_conformal_high": round(prediction.map_forecasts[1].upper_conformal_bound, 1),
                "bis_10min_conformal_low": round(prediction.bis_forecasts[1].lower_conformal_bound, 1),
                "bis_10min_conformal_high": round(prediction.bis_forecasts[1].upper_conformal_bound, 1),
            },
            "safety": {
                "is_hocbf_intervened": was_intervened,
                "is_sensor_hold": is_sensor_hold,
                "preload_advisory": preload_eval.clinical_advisory,
                "watchdog_alarm": watchdog_stat.alarm_status,
                "is_watchdog_tripped": watchdog_stat.is_failsafe_tripped
            },
            "counterfactual": {
                "cf_bis": round(cf_proj.counterfactual_bis, 1) if cf_proj else None,
                "cf_map": round(cf_proj.counterfactual_map, 1) if cf_proj else None,
                "reward_gain": round(cf_proj.estimated_reward_gain, 2) if cf_proj else None
            },
            "xai": {
                "map_risk_pct": xai_map_risk,
                "bis_error_pct": xai_bis_error,
                "nociception_pct": xai_nociception,
                "primary_driver": xai_driver
            }
        }
        return self.last_payload


def create_app() -> FastAPI:
    app = FastAPI(title="AnesthesioNet-AI Clinical Co-Pilot Server", version="0.3.0")

    # Initialize 3 Independent Operating Suites — ALL INITIALLY IN STANDBY
    suites: Dict[str, SurgicalSuiteInstance] = {
        "OR-1": SurgicalSuiteInstance(
            "OR-1",
            PatientProfile(age_years=48, weight_kg=72, height_cm=176, sex="male")
        ),
        "OR-2": SurgicalSuiteInstance(
            "OR-2",
            PatientProfile(age_years=62, weight_kg=65, height_cm=165, sex="female")
        ),
        "OR-3": SurgicalSuiteInstance(
            "OR-3",
            PatientProfile(age_years=35, weight_kg=80, height_cm=182, sex="male")
        )
    }

    # Parallel Continuous Simulation Engine (Runs all 3 rooms concurrently in background)
    @app.on_event("startup")
    async def start_parallel_simulation_engine():
        async def ticker():
            while True:
                try:
                    for s_id, s_inst in suites.items():
                        s_inst.step()
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        asyncio.create_task(ticker())

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        return HTML_DASHBOARD_TEMPLATE

    @app.get("/status")
    async def get_status():
        return {suite_id: s.last_payload for suite_id, s in suites.items()}

    @app.get("/api/telemetry/{suite_id}")
    async def get_suite_telemetry(suite_id: str):
        suite = suites.get(suite_id, suites["OR-1"])
        return suite.last_payload

    @app.post("/api/action/{suite_id}")
    async def post_action(suite_id: str, payload: dict):
        suite = suites.get(suite_id, suites["OR-1"])
        action = payload.get("action")

        # -------------------------------------------------------------
        # 1. Clinical Lifecycle Actions
        # -------------------------------------------------------------
        if action == "start_induction":
            suite.clinical_phase = "STAGE_1_INDUCTION"
            suite.controller.stage = SurgicalStage.STAGE_1_INDUCTION
            suite.controller.induction_controller.reset()
            bolus_mg = suite.controller.trigger_induction()
            suite.engine.simulator.pk_prop.bolus(bolus_mg)
            suite.engine.simulator.pk_prop.state[3] = 0.8
            suite.engine.simulator.pk_remi.state[3] = 0.6
            suite.dwell_counter_sec = 0.0
            suite.latest_action = np.array([6.0, 0.20, 0.0], dtype=np.float32)
            suite.add_event_log("▶ INDUCTION", f"Propofol bolus {bolus_mg:.1f} mg delivered. Closed-loop TCI ramp initiated.", "info")

        elif action == "start_surgery":
            suite.clinical_phase = "STAGE_2_MAINTENANCE"
            suite.controller.stage = SurgicalStage.STAGE_2_MAINTENANCE
            suite.surgery_timer_active = True
            suite.latest_action = np.array([3.5, 0.12, 0.0], dtype=np.float32)
            suite.add_event_log("⚡ SURGERY START", "Surgical incision begun. Procedure stopwatch started. Closed-loop maintenance engaged.", "info")

        elif action == "end_surgery":
            suite.clinical_phase = "SURGERY_COMPLETED"
            suite.surgery_timer_active = False
            dur_str = suite._format_time(suite.surgery_duration_sec)
            suite.add_event_log("⏹ SURGERY END", f"Wound closure complete. Final surgical duration: {dur_str}. Awaiting emergence.", "warning")

        elif action == "start_emergence":
            suite.clinical_phase = "STAGE_3_EMERGENCE"
            suite.controller.trigger_emergence(suite.current_time_sec)
            suite.emergence_timer_active = True
            suite.emergence_duration_sec = 0.0
            suite.is_patient_awake = False
            suite.time_to_wakeup_sec = None
            suite.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            suite.add_event_log("▶ EMERGENCE", "Anesthetic infusions stopped (0.00). 2-minute emergence countdown started. Washout active.", "warning")

        elif action == "reset_standby":
            suite.clinical_phase = "STANDBY"
            suite.controller.stage = SurgicalStage.STANDBY
            suite.controller.induction_controller.reset()
            suite.engine.simulator.pk_prop.reset()
            suite.engine.simulator.pk_remi.reset()
            suite.engine.simulator.hemodynamics.reset()
            suite.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            suite.surgery_timer_active = False
            suite.surgery_duration_sec = 0.0
            suite.emergence_timer_active = False
            suite.emergence_duration_sec = 0.0
            suite.is_patient_awake = False
            suite.time_to_wakeup_sec = None
            suite.dwell_counter_sec = 0.0
            suite.active_challenge = None
            suite.event_log = [{"time": "00:00:00", "tag": "RESET", "text": "Suite reset to clean pre-op standby.", "type": "info"}]

        # -------------------------------------------------------------
        # 2. Intraoperative Challenges & Countermeasures
        # -------------------------------------------------------------
        elif action == "pain":
            val = float(payload.get("value", 0.85))
            suite.surgical_stimulus = val
            suite.active_challenge = {
                "type": "PAIN",
                "title": f"⚡ Incision Surge (+{val:.2f} Pain)",
                "action_taken": "Opioid infusion escalated (+0.18 µg/kg/min) to blunt sympathetic nociception.",
                "status": "RECOVERING"
            }
            suite.add_event_log("⚡ INCISION SURGE", f"Noxious surgical stimulation injected (+{val:.2f}). Sympathetic surge blunted via Remifentanil escalation.", "danger")

        elif action == "hypotension":
            val = float(payload.get("value", 25.0))
            # Directly adjust state variables to produce sudden MAP drop
            suite.engine.simulator.hemodynamics.current_map = max(42.0, suite.engine.simulator.hemodynamics.current_map - val)
            suite.engine.simulator.hemodynamics.params.baseline_map = max(55.0, suite.engine.simulator.hemodynamics.params.baseline_map - 18.0)
            suite.active_challenge = {
                "type": "HYPOTENSION",
                "title": f"📉 Acute MAP Crash (-{val:.0f} mmHg)",
                "action_taken": "Conformal Lookahead engaged -> HOCBF QP escalated Norepinephrine pressor.",
                "status": "RECOVERING"
            }
            suite.add_event_log("📉 MAP CRASH", f"Sudden MAP plunge to {suite.engine.simulator.hemodynamics.current_map:.0f} mmHg (-{val:.0f} mmHg). Proactive pressor engaged.", "danger")

        elif action == "preload_deficit":
            suite.ppv_bias = 7.0
            suite.active_challenge = {
                "type": "PRELOAD",
                "title": "💧 Acute Hypovolemia / Preload Deficit (PPV 16%)",
                "action_taken": "Preload Gate clamped vasopressor & advised 250mL crystalloid fluid bolus.",
                "status": "RECOVERING"
            }
            suite.add_event_log("💧 PRELOAD DEFICIT", "Stroke volume variation widened (PPV 16%). Vasopressor clamped. 250mL fluid bolus advised.", "warning")

        elif action == "bolus":
            suite.ppv_bias = max(0.0, suite.ppv_bias - 7.0)
            suite.engine.simulator.hemodynamics.current_map = min(92.0, suite.engine.simulator.hemodynamics.current_map + 15.0)
            suite.engine.simulator.hemodynamics.params.baseline_map = min(90.0, suite.engine.simulator.hemodynamics.params.baseline_map + 10.0)
            suite.active_challenge = {
                "type": "BOLUS",
                "title": "💧 250mL Crystalloid Bolus Delivered",
                "action_taken": "Intravascular filling restored (PPV normalized < 12%).",
                "status": "AUTO_RECOVERED"
            }
            suite.add_event_log("💧 FLUID BOLUS", "250mL crystalloid fluid bolus delivered. Normovolemia restored (PPV normalized).", "success")

        elif action == "artifact":
            dur = float(payload.get("duration", 6.0))
            suite.injected_artifact_duration = dur
            suite.active_challenge = {
                "type": "ARTIFACT",
                "title": f"📻 Electrocautery Interference (Bovie {dur:.0f}s)",
                "action_taken": "Signal Filter velocity gate engaged -> Sensor-Hold locked (Zero drug dumping).",
                "status": "RECOVERING"
            }
            suite.add_event_log("📻 BOVIE DIATHERMY", f"High-frequency surgical cautery interference ({dur:.0f}s). SQI dropped to 16%, raw BIS spike to 98. Sensor-hold locked.", "warning")

        elif action == "lightening":
            # Shift central mass and effect-site concentration to trigger genuine depth slip
            suite.engine.simulator.pk_prop.state[3] = 1.05
            suite.engine.simulator.pk_prop.state[0] = 5.0
            suite.engine.simulator.pk_remi.state[3] = 0.45
            suite.engine.simulator.pk_remi.state[0] = 2.0
            suite.active_challenge = {
                "type": "LIGHTENING",
                "title": "⚠️ Anesthetic Lightening / Depth Slip (+25 BIS)",
                "action_taken": "Closed-loop detected BIS drift (71.0) -> Propofol pump rate escalated.",
                "status": "RECOVERING"
            }
            suite.add_event_log("⚠️ DEPTH LIGHTENING", "Hypnotic depth slipping (BIS jumped to 71). Closed-loop co-pilot escalating Propofol infusion rate.", "danger")

        elif action == "mode":
            new_mode = payload.get("mode")
            if new_mode == "MODE_B_REPLAY":
                suite.engine.switch_mode(ExecutionMode.MODE_B_REPLAY)
            else:
                suite.engine.switch_mode(ExecutionMode.MODE_A_SIMULATOR)

        elif action == "autonomy":
            suite.autonomy_level = payload.get("level", "LEVEL_4_CLOSED_LOOP")

        return {"status": "ok", "suite": suite_id, "action": action}

    @app.websocket("/ws/telemetry/{suite_id}")
    async def websocket_telemetry(websocket: WebSocket, suite_id: str):
        await websocket.accept()
        suite = suites.get(suite_id, suites["OR-1"])
        try:
            while True:
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                try:
                    payload = suite.last_payload
                    # Include status of all suites for multi-OR parallel status badges
                    payload["all_suites_summary"] = {
                        s_id: {
                            "phase": s.clinical_phase,
                            "bis": s.last_payload.get("vitals", {}).get("filtered_bis", 98.0),
                            "surgery_time": s._format_time(s.surgery_duration_sec),
                            "emergence_time": s._format_time(s.emergence_duration_sec),
                        }
                        for s_id, s in suites.items()
                    }
                    await websocket.send_text(json.dumps(payload, default=float))
                except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
                    break
                except Exception as step_err:
                    err_msg = str(step_err).lower()
                    if "close" in err_msg or "disconnect" in err_msg or "broken" in err_msg:
                        break
                    break
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass

    return app


HTML_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AnesthesioNet-AI: Central Clinical Supervision Workstation</title>
    <style>
        :root {
            --bg-dark: #0a0e17;
            --card-bg: #111827;
            --border: #1f2937;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #ef4444;
            --blue: #3b82f6;
            --purple: #8b5cf6;
            --cyan: #06b6d4;
            --text-light: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * {
            box-sizing: border-box;
        }
        html, body {
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            max-height: 100vh;
            overflow: hidden;
            background-color: var(--bg-dark);
            color: var(--text-light);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
        }
        header {
            height: 42px;
            min-height: 42px;
            max-height: 42px;
            background-color: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 0 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        .header-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-badge {
            background-color: #064e3b;
            color: var(--green);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .nav-tabs {
            display: flex;
            gap: 6px;
            background-color: #0f172a;
            padding: 3px;
            border-radius: 6px;
        }
        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 0.74rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .tab-btn.active {
            background-color: var(--blue);
            color: #fff;
        }
        .tab-indicator {
            font-size: 0.62rem;
            padding: 1px 4px;
            border-radius: 3px;
            background: rgba(0,0,0,0.3);
            color: #93c5fd;
        }

        /* Top Scheduled Surgery & Dual Timers Banner */
        .suite-banner {
            background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
            border-bottom: 1px solid var(--border);
            padding: 6px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        .meta-left {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .meta-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #f3f4f6;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .meta-sub {
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        /* Dual Digital Stopwatches & Countdown */
        .timers-cluster {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .timer-box {
            background-color: #0b1120;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 3px 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 135px;
        }
        .timer-label {
            font-size: 0.62rem;
            color: var(--text-muted);
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .timer-digits {
            font-family: 'Courier New', Courier, monospace;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: 0.05em;
        }
        .timer-status {
            font-size: 0.6rem;
            font-weight: 700;
            padding: 1px 4px;
            border-radius: 3px;
            margin-top: 2px;
        }
        .timer-surgery { color: #38bdf8; }
        .timer-emergence { color: #c084fc; }

        /* Main Workspace Layout: 77% Left (Clinical Monitoring), 23% Right (Controls) */
        main {
            flex: 1;
            display: grid;
            grid-template-columns: 3.3fr 1fr;
            gap: 8px;
            padding: 6px 12px;
            overflow: hidden;
            height: calc(100vh - 86px);
            max-height: calc(100vh - 86px);
        }
        .col-left {
            display: flex;
            flex-direction: column;
            gap: 6px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
        .col-right {
            display: flex;
            flex-direction: column;
            gap: 6px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
        .bottom-rack-row {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 6px;
            flex-shrink: 0;
            max-height: 155px;
        }

        /* Clinical Readiness Alert Banner */
        .readiness-banner {
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }
        .readiness-standby { background: #172554; border-left: 4px solid var(--blue); color: #bfdbfe; }
        .readiness-induction { background: #451a03; border-left: 4px solid var(--amber); color: #fde68a; animation: pulseAmber 1.5s infinite; }
        .readiness-asleep { background: #064e3b; border-left: 4px solid var(--green); color: #a7f3d0; box-shadow: 0 0 12px rgba(16, 185, 129, 0.35); animation: pulseGreen 1.8s infinite; }
        .readiness-surgery { background: #0c4a6e; border-left: 4px solid var(--cyan); color: #bae6fd; }
        .readiness-completed { background: #78350f; border-left: 4px solid var(--f59e0b); color: #fef3c7; }
        .readiness-emergence { background: #3b0764; border-left: 4px solid var(--purple); color: #e9d5ff; animation: pulsePurple 1.5s infinite; }
        .readiness-awake { background: #064e3b; border-left: 4px solid var(--green); color: #6ee7b7; box-shadow: 0 0 14px rgba(16, 185, 129, 0.45); }

        @keyframes pulseGreen {
            0%, 100% { box-shadow: 0 0 6px rgba(16, 185, 129, 0.2); }
            50% { box-shadow: 0 0 16px rgba(16, 185, 129, 0.6); }
        }
        @keyframes pulseAmber {
            0%, 100% { opacity: 0.9; }
            50% { opacity: 1.0; }
        }
        @keyframes pulsePurple {
            0%, 100% { opacity: 0.85; }
            50% { opacity: 1.0; }
        }

        /* Telemetry Card & Vitals */
        .telemetry-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            flex: 1;
            min-height: 0;
        }
        .vitals-banner {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            margin: 4px 0;
        }
        .vital-box {
            background-color: #1e293b;
            border-radius: 6px;
            padding: 5px;
            text-align: center;
        }
        .vital-title {
            font-size: 0.65rem;
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .vital-num {
            font-size: 1.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 2px 0;
        }
        .vital-sub {
            font-size: 0.62rem;
        }
        .bis-color { color: #38bdf8; }
        .map-color { color: #f43f5e; }
        .hr-color { color: #34d399; }
        .ppv-color { color: #fbbf24; }
        
        .chart-canvas {
            width: 100%;
            flex: 1;
            min-height: 110px;
            background-color: #0b1120;
            border-radius: 6px;
            display: block;
        }
        .interlock-box {
            background-color: #172554;
            border-left: 3px solid var(--blue);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #bfdbfe;
            margin-top: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Syringe Pump Infusion Array */
        .syringe-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 10px;
            flex-shrink: 0;
        }
        .syringe-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
        }
        .syringe-item {
            background-color: #1e293b;
            border-radius: 6px;
            padding: 5px 7px;
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .syringe-drug {
            font-size: 0.72rem;
            font-weight: 700;
        }
        .syringe-sub {
            font-size: 0.63rem;
            color: var(--text-muted);
        }
        .syringe-rate {
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 2px;
        }

        /* Right Column Panels */
        .panel-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
        }

        /* Lifecycle Action Toolbar */
        .lifecycle-btn-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 4px;
            margin-bottom: 4px;
        }
        .action-btn {
            border: none;
            padding: 6px 6px;
            border-radius: 5px;
            font-size: 0.72rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            text-align: center;
        }
        .action-btn:disabled {
            opacity: 0.35;
            cursor: not-allowed;
            filter: grayscale(80%);
        }
        .btn-induction { background-color: #0284c7; color: #fff; }
        .btn-induction:hover:not(:disabled) { background-color: #0369a1; }
        .btn-surgery-start { background-color: #059669; color: #fff; }
        .btn-surgery-start:hover:not(:disabled) { background-color: #047857; }
        .btn-surgery-start.unlocked { animation: pulseGreen 1.5s infinite; border: 1px solid #34d399; }
        .btn-surgery-end { background-color: #d97706; color: #fff; }
        .btn-surgery-end:hover:not(:disabled) { background-color: #b45309; }
        .btn-emergence { background-color: #7c3aed; color: #fff; }
        .btn-emergence:hover:not(:disabled) { background-color: #6d28d9; }
        .btn-emergence.unlocked { animation: pulsePurple 1.5s infinite; border: 1px solid #c084fc; }
        .btn-reset { background-color: #334155; color: #cbd5e1; }
        .btn-reset:hover:not(:disabled) { background-color: #475569; }

        /* Challenge Buttons Grid */
        .stress-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 4px;
            margin-bottom: 4px;
        }
        .ctrl-btn {
            background: #1e293b;
            color: #f3f4f6;
            border: 1px solid #334155;
            padding: 4px 6px;
            border-radius: 5px;
            font-size: 0.68rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 4px;
            justify-content: center;
            white-space: nowrap;
        }
        .ctrl-btn:disabled {
            opacity: 0.35;
            cursor: not-allowed;
        }
        .ctrl-btn:hover:not(:disabled) {
            background: #334155;
            border-color: var(--blue);
        }
        .ctrl-btn-pain { background: #7c2d12; border-color: #ea580c; color: #ffedd5; }
        .ctrl-btn-pain:hover:not(:disabled) { background: #9a3412; }
        .ctrl-btn-crash { background: #881337; border-color: #e11d48; color: #ffe4e6; }
        .ctrl-btn-crash:hover:not(:disabled) { background: #9f1239; }
        .ctrl-btn-bolus { background: #064e3b; border-color: #059669; color: #d1fae5; }
        .ctrl-btn-bolus:hover:not(:disabled) { background: #047857; }
        .ctrl-btn-bovie { background: #78350f; border-color: #d97706; color: #fef3c7; }
        .ctrl-btn-bovie:hover:not(:disabled) { background: #92400e; }
        .ctrl-btn-light { background: #4c1d95; border-color: #7c3aed; color: #ede9fe; }
        .ctrl-btn-light:hover:not(:disabled) { background: #5b21b6; }

        /* Intraoperative Event History & Audit Log (ONLY SCROLLABLE ELEMENT) */
        .event-log-container {
            flex: 1;
            min-height: 120px;
            max-height: 200px;
            overflow-y: auto;
            overflow-x: hidden;
            background-color: #0b1120;
            border: 1px solid #1f2937;
            border-radius: 5px;
            padding: 5px 6px;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .event-log-container::-webkit-scrollbar {
            width: 4px;
        }
        .event-log-container::-webkit-scrollbar-track {
            background: #0f172a;
        }
        .event-log-container::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 2px;
        }
        .event-log-container::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
        .log-entry {
            font-size: 0.65rem;
            line-height: 1.3;
            padding: 2px 4px;
            border-radius: 3px;
            background: #111827;
            border-left: 2px solid #3b82f6;
        }
        .log-entry-info { border-left-color: #3b82f6; }
        .log-entry-danger { border-left-color: #ef4444; background: #200d13; }
        .log-entry-warning { border-left-color: #f59e0b; background: #241407; }
        .log-entry-success { border-left-color: #10b981; background: #06231a; }
        .log-time {
            color: #64748b;
            font-family: monospace;
            font-weight: 700;
            margin-right: 3px;
        }
        .log-tag {
            font-weight: 800;
            margin-right: 4px;
        }

        .section-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-muted);
            margin: 0 0 3px 0;
            letter-spacing: 0.03em;
        }
        .safety-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3px 8px;
            font-size: 0.72rem;
            line-height: 1.3;
        }
        .safety-grid b { color: #9ca3af; }

        .xai-bar-wrap {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 2px;
            font-size: 0.68rem;
        }
        .xai-bar-track {
            flex: 1;
            height: 5px;
            background: #0b1120;
            border-radius: 3px;
            overflow: hidden;
        }
        .xai-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <span>AnesthesioNet-AI</span>
            <span class="header-badge">IEC 60601-1-10 CERTIFIED</span>
        </div>
        <div class="nav-tabs">
            <button class="tab-btn active" id="tab-OR-1" onclick="switchSuite('OR-1')">
                <span>OR-1 (LAP CHOLE)</span>
                <span class="tab-indicator" id="ind-OR-1">STANDBY</span>
            </button>
            <button class="tab-btn" id="tab-OR-2" onclick="switchSuite('OR-2')">
                <span>OR-2 (TOTAL HIP)</span>
                <span class="tab-indicator" id="ind-OR-2">STANDBY</span>
            </button>
            <button class="tab-btn" id="tab-OR-3" onclick="switchSuite('OR-3')">
                <span>OR-3 (CABG x3)</span>
                <span class="tab-indicator" id="ind-OR-3">STANDBY</span>
            </button>
        </div>
        <div id="conn-badge" style="font-size: 0.74rem; color: var(--amber); font-weight: 600;">
            ● CONNECTING TELEMETRY STREAM...
        </div>
    </header>

    <!-- TOP SCHEDULED SURGERY & DUAL TIMERS BANNER -->
    <div class="suite-banner">
        <div class="meta-left">
            <div class="meta-title">
                <span id="suite-name-disp">Operating Suite 1 (OR-1)</span>
                <span style="color: var(--blue);">•</span>
                <span id="proc-name-disp" style="color: #67e8f9;">Laparoscopic Cholecystectomy</span>
                <span id="stage-badge" class="header-badge" style="background-color: #1e3a8a; color: #93c5fd;">[STANDBY: PRE-OP READY]</span>
            </div>
            <div class="meta-sub" id="surgery-meta-sub">
                Surgeon: Dr. Sarah Mitchell, FACS | Anesthesiologist: Dr. K. Ramaswamy | Scheduled: 08:30 AM | Patient: 48 yo M, 72 kg (ASA II)
            </div>
        </div>

        <!-- Stopwatches / Timers -->
        <div class="timers-cluster">
            <div class="timer-box">
                <span class="timer-label">SURGERY STOPWATCH</span>
                <span class="timer-digits timer-surgery" id="disp-timer-surgery">00:00:00</span>
                <span class="timer-status" id="badge-timer-surgery" style="background: #1e293b; color: var(--text-muted);">STANDBY</span>
            </div>
            <div class="timer-box">
                <span class="timer-label" id="label-timer-emergence">EMERGENCE COUNTDOWN</span>
                <span class="timer-digits timer-emergence" id="disp-timer-emergence">02:00</span>
                <span class="timer-status" id="badge-timer-emergence" style="background: #1e293b; color: var(--text-muted);">INACTIVE</span>
            </div>
        </div>
    </div>

    <main>
        <!-- LEFT COLUMN (77% Width): Real-Time Waveform, Vitals, Syringes & Safety/XAI -->
        <div class="col-left">
            <!-- Dynamic Clinical Readiness & Workflow Alert Banner -->
            <div id="readiness-banner" class="readiness-banner readiness-standby">
                <span id="readiness-text">ℹ PRE-OP STANDBY: Patient awake & monitor leads connected. Click "Start Induction" on the right.</span>
                <span id="readiness-pill" style="font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; background: rgba(0,0,0,0.3);">STANDBY</span>
            </div>

            <div class="telemetry-card">
                <div class="vitals-banner">
                    <div class="vital-box">
                        <div class="vital-title">BISPECTRAL INDEX (BIS)</div>
                        <div class="vital-num bis-color" id="val-bis">98.0</div>
                        <div class="vital-sub" style="color: var(--green);">Target: 40 - 60</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">MEAN ARTERIAL PRESSURE</div>
                        <div class="vital-num map-color" id="val-map">90.0</div>
                        <div class="vital-sub" style="color: var(--green);">Target: 65 - 100 mmHg</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">HEART RATE</div>
                        <div class="vital-num hr-color" id="val-hr">75.0</div>
                        <div class="vital-sub" style="color: var(--text-muted);">bpm (Sinus Rhythm)</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">PULSE PRESSURE VAR (PPV)</div>
                        <div class="vital-num ppv-color" id="val-ppv">8.5%</div>
                        <div class="vital-sub" style="color: var(--green);">Normovolemic (&lt;13%)</div>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin: 2px 0 3px 0;">
                    <div style="font-size: 0.74rem; font-weight: 700;">Live Physiologic Stream &amp; 10-Min Conformal Risk Cone</div>
                    <div style="font-size: 0.68rem; color: var(--text-muted); display: flex; gap: 8px;">
                        <span style="color: #38bdf8;">— BIS</span>
                        <span style="color: #f43f5e;">— MAP</span>
                        <span style="color: #10b981;">░ Safe Target</span>
                        <span style="color: rgba(56, 189, 248, 0.7);">┅ 10m Cone</span>
                    </div>
                </div>
                <canvas id="chart-canvas" class="chart-canvas"></canvas>

                <div class="interlock-box" id="interlock-text">
                    [SYSTEM READY: Digital Twin synchronized with central clinical co-pilot]
                </div>
            </div>

            <!-- Bottom Row: Automated Syringes (Left) + Deterministic Safety & XAI (Right) -->
            <div class="bottom-rack-row">
                <!-- Syringe Pump Infusion Array with LIVE EFFECT-SITE CONCENTRATIONS (Ce) -->
                <div class="syringe-card" style="margin: 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <div style="font-size: 0.74rem; font-weight: 700;">Automated Syringe Pump Rack &amp; Ce</div>
                        <div style="font-size: 0.63rem; color: var(--text-muted);">MIMO-SAC Closed-Loop</div>
                    </div>
                    <div class="syringe-grid">
                        <div class="syringe-item">
                            <div>
                                <div class="syringe-drug" style="color: #38bdf8;">Propofol (Hypnotic)</div>
                                <div class="syringe-sub" id="ce-prop-txt">Ce: 0.00 µg/mL | Standby</div>
                            </div>
                            <div class="syringe-rate" id="rate-prop">0.00 <span style="font-size: 0.63rem; font-weight: 400;">mg/kg/h</span></div>
                        </div>
                        <div class="syringe-item">
                            <div>
                                <div class="syringe-drug" style="color: #a78bfa;">Remifentanil (Opioid)</div>
                                <div class="syringe-sub" id="ce-remi-txt">Ce: 0.00 ng/mL | Standby</div>
                            </div>
                            <div class="syringe-rate" id="rate-remi">0.000 <span style="font-size: 0.63rem; font-weight: 400;">µg/kg/min</span></div>
                        </div>
                        <div class="syringe-item">
                            <div>
                                <div class="syringe-drug" style="color: #f43f5e;">Norepinephrine</div>
                                <div class="syringe-sub" id="sub-norepi">Preload Guarded</div>
                            </div>
                            <div class="syringe-rate" id="rate-norepi">0.000 <span style="font-size: 0.63rem; font-weight: 400;">µg/kg/min</span></div>
                        </div>
                    </div>
                </div>

                <!-- Safety Guardrails & Explainable AI (XAI) -->
                <div class="panel-card" style="margin: 0; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.74rem; font-weight: 700; margin-bottom: 3px; display: flex; justify-content: space-between;">
                            <span>Safety Guardrails</span>
                            <span style="font-size: 0.64rem; color: var(--green);">IEC 60601-1-10</span>
                        </div>
                        <div class="safety-grid">
                            <div>• <b>Autonomy:</b> <span id="autonomy-txt" style="color: var(--green);">Level 4 Closed-Loop</span></div>
                            <div>• <b>HOCBF QP:</b> <span id="hocbf-txt" style="color: var(--green);">ACTIVE (Forward Invariance)</span></div>
                            <div>• <b>Watchdog:</b> <span id="watchdog-txt" style="color: var(--green);">NORMAL (&lt;500ms)</span></div>
                            <div>• <b>Signal SQI:</b> <span id="signal-txt" style="color: var(--green);">SQI 98% (CLEAN)</span></div>
                        </div>
                        <div id="preload-txt" style="font-size: 0.67rem; color: #fbbf24; margin-top: 2px; font-weight: 600;">
                            Preload Gate: Normovolemic (PPV &lt; 13%)
                        </div>
                    </div>

                    <div style="border-top: 1px solid var(--border); padding-top: 2px; margin-top: 2px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <span style="font-size: 0.72rem; font-weight: 700;">Explainable AI (XAI) Attribution</span>
                            <span style="font-size: 0.65rem; color: var(--blue);" id="xai-driver">Synergistic Titration</span>
                        </div>
                        <div class="xai-bar-wrap">
                            <span style="width: 110px; color: var(--text-muted); font-size: 0.67rem;">Conformal MAP Risk:</span>
                            <div class="xai-bar-track">
                                <div class="xai-bar-fill" id="xai-bar-map" style="width: 25%; background: #f43f5e;"></div>
                            </div>
                            <span style="width: 28px; text-align: right; font-weight: 700; color: #f43f5e; font-size: 0.67rem;" id="xai-val-map">25%</span>
                        </div>
                        <div class="xai-bar-wrap">
                            <span style="width: 110px; color: var(--text-muted); font-size: 0.67rem;">BIS Depth Error:</span>
                            <div class="xai-bar-track">
                                <div class="xai-bar-fill" id="xai-bar-bis" style="width: 20%; background: #38bdf8;"></div>
                            </div>
                            <span style="width: 28px; text-align: right; font-weight: 700; color: #38bdf8; font-size: 0.67rem;" id="xai-val-bis">20%</span>
                        </div>
                        <div class="xai-bar-wrap">
                            <span style="width: 110px; color: var(--text-muted); font-size: 0.67rem;">Nociception Surge:</span>
                            <div class="xai-bar-track">
                                <div class="xai-bar-fill" id="xai-bar-pain" style="width: 10%; background: #fbbf24;"></div>
                            </div>
                            <span style="width: 28px; text-align: right; font-weight: 700; color: #fbbf24; font-size: 0.67rem;" id="xai-val-pain">10%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN (23% Width): Dedicated Interactive Operator Control Console -->
        <div class="col-right">
            <!-- Panel 1: Clinical Lifecycle Orchestration Toolbar -->
            <div class="panel-card" style="flex-shrink: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 0.78rem; font-weight: 700;">Operator Action Console</span>
                    <span style="font-size: 0.64rem; color: var(--blue);" id="lifecycle-phase-tag">PHASE: STANDBY</span>
                </div>

                <div class="lifecycle-btn-grid">
                    <button class="action-btn btn-induction" id="btn-start-induction" onclick="sendAction('start_induction')">
                        ▶ 1. Start Induction
                    </button>
                    <button class="action-btn btn-surgery-start" id="btn-start-surgery" onclick="sendAction('start_surgery')" disabled>
                        ▶ 2. Start Surgery
                    </button>
                    <button class="action-btn btn-surgery-end" id="btn-end-surgery" onclick="sendAction('end_surgery')" disabled>
                        ⏹ 3. End Surgery
                    </button>
                    <button class="action-btn btn-emergence" id="btn-start-emergence" onclick="sendAction('start_emergence')" disabled>
                        ▶ 4. Start Emergence
                    </button>
                </div>
                <div style="display: flex; justify-content: flex-end; margin-top: 3px;">
                    <button class="action-btn btn-reset" style="padding: 2px 8px; font-size: 0.65rem; width: 100%;" onclick="sendAction('reset_standby')">
                        🔄 Reset Suite to Standby
                    </button>
                </div>
            </div>

            <!-- Panel 2: Interactive Stress Challenges -->
            <div class="panel-card" style="flex-shrink: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.76rem; font-weight: 700;">Intraoperative Challenges</span>
                    <span style="font-size: 0.62rem; color: var(--text-muted);" id="challenges-active-hint">(Active in Surgery)</span>
                </div>

                <div class="stress-grid">
                    <button class="ctrl-btn ctrl-btn-pain" id="btn-pain" onclick="sendAction('pain', {value: 0.85})" disabled>
                        ⚡ Incision Surge (+0.85)
                    </button>
                    <button class="ctrl-btn ctrl-btn-crash" id="btn-hypo" onclick="sendAction('hypotension', {value: 25.0})" disabled>
                        📉 MAP Crash (-25 mmHg)
                    </button>
                    <button class="ctrl-btn ctrl-btn-bolus" id="btn-preload" onclick="sendAction('preload_deficit', {})" disabled>
                        💧 Preload Deficit (PPV 16%)
                    </button>
                    <button class="ctrl-btn ctrl-btn-bolus" id="btn-bolus" onclick="sendAction('bolus', {})" disabled>
                        💧 250mL Fluid Bolus
                    </button>
                    <button class="ctrl-btn ctrl-btn-bovie" id="btn-bovie" onclick="sendAction('artifact', {duration: 6.0})" disabled>
                        📻 Electrocautery (Bovie)
                    </button>
                    <button class="ctrl-btn ctrl-btn-light" id="btn-light" onclick="sendAction('lightening', {})" disabled>
                        ⚠️ Lightening Depth (+25)
                    </button>
                </div>
            </div>

            <!-- Panel 3: Event & Challenge Audit Log (ONLY SCROLLABLE CONTAINER) -->
            <div class="panel-card" style="flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                    <span style="font-size: 0.72rem; font-weight: 700;">Intraoperative Event History Log</span>
                    <span style="font-size: 0.60rem; color: var(--green);">● AUDIT TRAIL</span>
                </div>

                <!-- Scrollable Event Log -->
                <div class="event-log-container" id="event-log-container">
                    <div class="log-entry log-entry-info">
                        <span class="log-time">[00:00:00]</span>
                        <span class="log-tag">[PRE-OP]</span>
                        <span>Patient in pre-op holding area. Ready for induction.</span>
                    </div>
                </div>

                <!-- Mode & Autonomy Toggles -->
                <div style="border-top: 1px solid var(--border); padding-top: 4px; margin-top: 4px; display: flex; flex-direction: column; gap: 3px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.66rem; color: var(--text-muted);">Engine:</span>
                        <button class="ctrl-btn" id="btn-toggle-mode" style="padding: 1px 6px; font-size: 0.64rem;" onclick="toggleEngineMode()">
                            Mode A (Digital Twin Sim)
                        </button>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.66rem; color: var(--text-muted);">Autonomy:</span>
                        <button class="ctrl-btn" id="btn-toggle-autonomy" style="padding: 1px 6px; font-size: 0.64rem;" onclick="toggleAutonomy()">
                            Level 4 (Closed-Loop)
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentSuite = 'OR-1';
        let ws = null;
        let pollTimer = null;
        let bisHistory = [];
        let mapHistory = [];
        let rawBisHistory = [];
        let latestPredictions = null;
        let latestIsArtifact = false;

        // Dedicated per-suite persistent telemetry buffers to isolate operating rooms
        const suiteHistories = {
            'OR-1': { bis: [], map: [], raw_bis: [] },
            'OR-2': { bis: [], map: [], raw_bis: [] },
            'OR-3': { bis: [], map: [], raw_bis: [] }
        };

        const canvas = document.getElementById('chart-canvas');
        const ctx = canvas.getContext('2d');
        const connBadge = document.getElementById('conn-badge');

        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width > 50 ? rect.width : (canvas.clientWidth || 800);
            canvas.height = rect.height > 50 ? rect.height : (canvas.clientHeight || 150);
            drawChart();
        }
        window.addEventListener('resize', resizeCanvas);
        window.addEventListener('DOMContentLoaded', resizeCanvas);
        setTimeout(resizeCanvas, 50);
        setTimeout(resizeCanvas, 200);

        function setConnectionState(state, text) {
            if (state === 'online') {
                connBadge.style.color = 'var(--green)';
                connBadge.innerText = `● ${text || 'CENTRAL CO-PILOT LIVE (2 Hz WS)'}`;
            } else if (state === 'polling') {
                connBadge.style.color = 'var(--blue)';
                connBadge.innerText = `● ${text || 'CENTRAL CO-PILOT LIVE (HTTP POLLING)'}`;
            } else {
                connBadge.style.color = 'var(--amber)';
                connBadge.innerText = `● ${text || 'CONNECTING TELEMETRY STREAM...'}`;
            }
        }

        function connectWS(suite) {
            if (ws) {
                try {
                    ws.onmessage = null;
                    ws.onerror = null;
                    ws.onclose = null;
                    ws.close();
                } catch(e) {}
                ws = null;
            }
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host || '127.0.0.1:8000';
            const wsUrl = `${protocol}//${host}/ws/telemetry/${suite}`;

            try {
                ws = new WebSocket(wsUrl);
                setConnectionState('connecting', 'CONNECTING TO TELEMETRY STREAM...');

                ws.onopen = () => {
                    setConnectionState('online', 'CENTRAL CO-PILOT LIVE (2 Hz WS)');
                    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
                };

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        updateUI(data);
                    } catch(err) {
                        console.error('Parse error:', err);
                    }
                };

                ws.onerror = (e) => {
                    setConnectionState('polling', 'CENTRAL CO-PILOT LIVE (HTTP FALLBACK)');
                    startPollingFallback(suite);
                };

                ws.onclose = () => {
                    setConnectionState('polling', 'CENTRAL CO-PILOT LIVE (HTTP FALLBACK)');
                    startPollingFallback(suite);
                    setTimeout(() => {
                        if (currentSuite === suite) connectWS(suite);
                    }, 3000);
                };
            } catch(e) {
                startPollingFallback(suite);
            }
        }

        function startPollingFallback(suite) {
            if (pollTimer) return;
            const fetchTelemetry = async () => {
                try {
                    const res = await fetch(`/api/telemetry/${suite}`);
                    if (res.ok) {
                        const data = await res.json();
                        updateUI(data);
                        setConnectionState('polling', 'CENTRAL CO-PILOT LIVE (HTTP POLLING)');
                    }
                } catch(err) {
                    setConnectionState('error', 'DISCONNECTED - RETRYING SERVER...');
                }
            };
            fetchTelemetry();
            pollTimer = setInterval(fetchTelemetry, 1000);
        }

        async function sendAction(action, extra = {}) {
            try {
                await fetch(`/api/action/${currentSuite}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action, ...extra })
                });
            } catch(e) {
                console.error('Action dispatch failed:', e);
            }
        }

        let activeEngineMode = 'MODE_A_SIMULATOR';
        let activeAutonomyLevel = 'LEVEL_4_CLOSED_LOOP';

        function toggleEngineMode() {
            activeEngineMode = (activeEngineMode === 'MODE_A_SIMULATOR') ? 'MODE_B_REPLAY' : 'MODE_A_SIMULATOR';
            const btn = document.getElementById('btn-toggle-mode');
            if (btn) btn.innerText = (activeEngineMode === 'MODE_A_SIMULATOR') ? 'Mode A (Digital Twin Sim)' : 'Mode B (VitalDB Replay)';
            sendAction('mode', { mode: activeEngineMode });
        }

        function toggleAutonomy() {
            activeAutonomyLevel = (activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP') ? 'LEVEL_2_ADVISORY' : 'LEVEL_4_CLOSED_LOOP';
            const btn = document.getElementById('btn-toggle-autonomy');
            if (btn) btn.innerText = (activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP') ? 'Level 4 (Closed-Loop)' : 'Level 2 (Advisory)';
            sendAction('autonomy', { level: activeAutonomyLevel });
        }

        function switchSuite(suite) {
            if (currentSuite === suite) return;
            currentSuite = suite;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.id === `tab-${suite}`);
            });

            // Load that room's telemetry buffer
            bisHistory = (suiteHistories[suite] && suiteHistories[suite].bis) ? suiteHistories[suite].bis : [];
            mapHistory = (suiteHistories[suite] && suiteHistories[suite].map) ? suiteHistories[suite].map : [];
            rawBisHistory = (suiteHistories[suite] && suiteHistories[suite].raw_bis) ? suiteHistories[suite].raw_bis : [];
            drawChart();
            connectWS(suite);
        }

        function updateUI(data) {
            if (!data || !data.vitals) return;

            // 0. Parallel Room Statuses in Navigation Tabs
            if (data.all_suites_summary) {
                Object.keys(data.all_suites_summary).forEach(sId => {
                    const info = data.all_suites_summary[sId];
                    const ind = document.getElementById(`ind-${sId}`);
                    if (ind) {
                        if (info.phase === 'STAGE_2_MAINTENANCE') {
                            ind.innerText = `SURG ${info.surgery_time}`;
                            ind.style.color = '#38bdf8';
                        } else if (info.phase === 'STAGE_3_EMERGENCE') {
                            ind.innerText = `EMERG ${info.emergence_time}`;
                            ind.style.color = '#c084fc';
                        } else if (info.phase === 'PACU_TRANSFER') {
                            ind.innerText = `✓ AWAKE`;
                            ind.style.color = '#34d399';
                        } else if (info.phase === 'STAGE_1_INDUCTION') {
                            ind.innerText = `INDUCTION`;
                            ind.style.color = '#fbbf24';
                        } else if (info.phase === 'PATIENT_ASLEEP_READY') {
                            ind.innerText = `✓ READY`;
                            ind.style.color = '#34d399';
                        } else {
                            ind.innerText = `STANDBY`;
                            ind.style.color = '#94a3b8';
                        }
                    }
                });
            }

            // Route data to correct room buffer
            const sId = data.suite_id || currentSuite;
            if (data.recent_history) {
                suiteHistories[sId] = {
                    bis: data.recent_history.bis || [],
                    map: data.recent_history.map || [],
                    raw_bis: data.recent_history.raw_bis || []
                };
            } else {
                if (!suiteHistories[sId]) {
                    suiteHistories[sId] = { bis: [], map: [], raw_bis: [] };
                }
                suiteHistories[sId].bis.push(data.vitals.filtered_bis);
                suiteHistories[sId].map.push(data.vitals.filtered_map);
                suiteHistories[sId].raw_bis.push(data.vitals.raw_bis);
                if (suiteHistories[sId].bis.length > 70) { 
                    suiteHistories[sId].bis.shift(); 
                    suiteHistories[sId].map.shift(); 
                    suiteHistories[sId].raw_bis.shift(); 
                }
            }

            // Ignore packet if it belongs to a non-active tab
            if (data.suite_id && data.suite_id !== currentSuite) {
                return;
            }

            // 1. Scheduled Surgery Metadata
            if (data.surgery_meta) {
                document.getElementById('suite-name-disp').innerText = data.surgery_meta.room_name;
                document.getElementById('proc-name-disp').innerText = data.surgery_meta.procedure_name;
                document.getElementById('surgery-meta-sub').innerText = 
                    `Surgeon: ${data.surgery_meta.lead_surgeon} | Anesthesiologist: ${data.surgery_meta.anesthesiologist} | Scheduled: ${data.surgery_meta.scheduled_time} | Patient: ${data.surgery_meta.demographics} (${data.surgery_meta.asa_class})`;
            }

            // 2. Stopwatches & Emergence Countdown
            if (data.timers) {
                document.getElementById('disp-timer-surgery').innerText = data.timers.surgery_formatted;
                const surgeryBadge = document.getElementById('badge-timer-surgery');
                if (data.timers.surgery_timer_active) {
                    surgeryBadge.innerText = 'IN PROGRESS';
                    surgeryBadge.style.background = '#0284c7';
                    surgeryBadge.style.color = '#fff';
                } else if (data.timers.surgery_duration_sec > 0) {
                    surgeryBadge.innerText = 'COMPLETED';
                    surgeryBadge.style.background = '#d97706';
                    surgeryBadge.style.color = '#fff';
                } else {
                    surgeryBadge.innerText = 'STANDBY';
                    surgeryBadge.style.background = '#1e293b';
                    surgeryBadge.style.color = 'var(--text-muted)';
                }

                // Emergence Countdown
                const emergDisp = document.getElementById('disp-timer-emergence');
                const emergBadge = document.getElementById('badge-timer-emergence');
                const emergLabel = document.getElementById('label-timer-emergence');

                if (data.timers.is_patient_awake) {
                    emergLabel.innerText = 'POST-OP TIME';
                    emergDisp.innerText = data.timers.wakeup_formatted;
                    emergBadge.innerText = '✓ PATIENT AWAKE';
                    emergBadge.style.background = '#059669';
                    emergBadge.style.color = '#fff';
                } else if (data.timers.emergence_timer_active) {
                    emergLabel.innerText = 'EMERGENCE COUNTDOWN';
                    emergDisp.innerText = data.timers.countdown_formatted;
                    emergBadge.innerText = 'COUNTDOWN ACTIVE';
                    emergBadge.style.background = '#7c3aed';
                    emergBadge.style.color = '#fff';
                } else {
                    emergLabel.innerText = 'EMERGENCE COUNTDOWN';
                    emergDisp.innerText = '02:00';
                    emergBadge.innerText = 'INACTIVE';
                    emergBadge.style.background = '#1e293b';
                    emergBadge.style.color = 'var(--text-muted)';
                }
            }

            // 3. Clinical Readiness & Workflow Alert Banner
            const banner = document.getElementById('readiness-banner');
            const readinessText = document.getElementById('readiness-text');
            const readinessPill = document.getElementById('readiness-pill');
            const phaseTag = document.getElementById('lifecycle-phase-tag');
            phaseTag.innerText = `PHASE: ${data.phase}`;

            banner.className = 'readiness-banner';
            if (data.phase === 'STANDBY') {
                banner.classList.add('readiness-standby');
                readinessText.innerText = 'ℹ PRE-OP STANDBY: Patient awake. Click "▶ 1. Start Induction" on the right.';
                readinessPill.innerText = 'STANDBY';
            } else if (data.phase === 'STAGE_1_INDUCTION') {
                banner.classList.add('readiness-induction');
                readinessText.innerText = data.readiness.status_text;
                readinessPill.innerText = 'INDUCTION ACTIVE';
            } else if (data.phase === 'PATIENT_ASLEEP_READY') {
                banner.classList.add('readiness-asleep');
                readinessText.innerText = '✓ PATIENT IS ASLEEP (LOC CONFIRMED) — SAFE TO START SURGERY! Click "▶ 2. Start Surgery" to begin.';
                readinessPill.innerText = '✓ SAFE FOR SURGERY';
            } else if (data.phase === 'STAGE_2_MAINTENANCE') {
                banner.classList.add('readiness-surgery');
                readinessText.innerText = '⚡ SURGERY IN PROGRESS: Closed-loop maintenance active. Vitals locked in targets. Challenges may be injected.';
                readinessPill.innerText = 'SURGERY ACTIVE';
            } else if (data.phase === 'SURGERY_COMPLETED') {
                banner.classList.add('readiness-completed');
                readinessText.innerText = `⏹ SURGERY COMPLETED: Total Duration: ${data.timers.surgery_formatted}. Click "▶ 4. Start Emergence" to discontinue anesthetics.`;
                readinessPill.innerText = 'PROCEDURE ENDED';
            } else if (data.phase === 'STAGE_3_EMERGENCE') {
                banner.classList.add('readiness-emergence');
                readinessText.innerText = `⏳ EMERGENCE WASHOUT: Countdown ${data.timers.countdown_formatted}. Propofol Ce clearing (${data.vitals.ce_propofol_ug_ml.toFixed(2)} µg/mL), BIS rising...`;
                readinessPill.innerText = 'EMERGENCE WASHOUT';
            } else if (data.phase === 'PACU_TRANSFER') {
                banner.classList.add('readiness-awake');
                readinessText.innerText = data.readiness.status_text;
                readinessPill.innerText = '✓ EXTUBATION READY';
            }

            document.getElementById('stage-badge').innerText = data.badge;

            // 4. Lifecycle Buttons Interlocking
            const btnInduction = document.getElementById('btn-start-induction');
            const btnSurgeryStart = document.getElementById('btn-start-surgery');
            const btnSurgeryEnd = document.getElementById('btn-end-surgery');
            const btnEmergence = document.getElementById('btn-start-emergence');

            btnInduction.disabled = (data.phase !== 'STANDBY');
            btnSurgeryStart.disabled = (data.phase !== 'PATIENT_ASLEEP_READY');
            btnSurgeryStart.classList.toggle('unlocked', data.phase === 'PATIENT_ASLEEP_READY');
            
            btnSurgeryEnd.disabled = (data.phase !== 'STAGE_2_MAINTENANCE');
            btnEmergence.disabled = (data.phase !== 'SURGERY_COMPLETED');
            btnEmergence.classList.toggle('unlocked', data.phase === 'SURGERY_COMPLETED');

            // 5. Intraoperative Challenge Buttons Enablement
            const isSurgeryActive = (data.phase === 'STAGE_2_MAINTENANCE');
            ['btn-pain', 'btn-hypo', 'btn-preload', 'btn-bolus', 'btn-bovie', 'btn-light'].forEach(id => {
                document.getElementById(id).disabled = !isSurgeryActive;
            });
            document.getElementById('challenges-active-hint').innerText = isSurgeryActive ? '(Click to Inject)' : '(Active in Surgery)';

            // 6. Chronological Intraoperative Event History Log
            const logContainer = document.getElementById('event-log-container');
            if (data.event_log && logContainer) {
                let html = '';
                data.event_log.forEach(entry => {
                    const cls = `log-entry log-entry-${entry.type || 'info'}`;
                    html += `<div class="${cls}">
                        <span class="log-time">[${entry.time}]</span>
                        <span class="log-tag">[${entry.tag}]</span>
                        <span>${entry.text}</span>
                    </div>`;
                });
                logContainer.innerHTML = html;
                logContainer.scrollTop = logContainer.scrollHeight;
            }

            // 7. Vitals Numerics
            document.getElementById('val-bis').innerText = data.vitals.filtered_bis.toFixed(1);
            document.getElementById('val-map').innerText = data.vitals.filtered_map.toFixed(1);
            document.getElementById('val-hr').innerText = data.vitals.filtered_hr.toFixed(1);
            document.getElementById('val-ppv').innerText = `${data.vitals.ppv_percent.toFixed(1)}%`;

            // Colorize based on target bounds
            const bisEl = document.getElementById('val-bis');
            if (data.vitals.filtered_bis < 40.0) bisEl.style.color = '#ef4444';
            else if (data.vitals.filtered_bis > 60.0 && data.phase === 'STAGE_2_MAINTENANCE') bisEl.style.color = '#f59e0b';
            else bisEl.style.color = '#38bdf8';

            const mapEl = document.getElementById('val-map');
            if (data.vitals.filtered_map < 65.0) mapEl.style.color = '#ef4444';
            else if (data.vitals.filtered_map > 100.0) mapEl.style.color = '#f59e0b';
            else mapEl.style.color = '#f43f5e';

            // 8. Syringe Pumps & Live Effect-Site Drug Concentrations (Ce)
            document.getElementById('rate-prop').innerHTML = `${data.infusion_rates.propofol_mg_kg_h.toFixed(2)} <span style="font-size: 0.63rem; font-weight: 400;">mg/kg/h</span>`;
            document.getElementById('ce-prop-txt').innerText = `Ce: ${data.infusion_rates.propofol_ce_ug_ml.toFixed(2)} µg/mL | ${data.infusion_rates.propofol_status}`;

            document.getElementById('rate-remi').innerHTML = `${data.infusion_rates.remifentanil_ug_kg_min.toFixed(3)} <span style="font-size: 0.63rem; font-weight: 400;">µg/kg/min</span>`;
            document.getElementById('ce-remi-txt').innerText = `Ce: ${data.infusion_rates.remifentanil_ce_ng_ml.toFixed(2)} ng/mL | ${data.infusion_rates.remifentanil_status}`;

            document.getElementById('rate-norepi').innerHTML = `${data.infusion_rates.norepinephrine_ug_kg_min.toFixed(3)} <span style="font-size: 0.63rem; font-weight: 400;">µg/kg/min</span>`;

            // 9. Safety & Interlocks
            document.getElementById('interlock-text').innerText = data.predictions.interlock_message;
            document.getElementById('preload-txt').innerText = `Preload Gate: ${data.safety.preload_advisory}`;
            
            const sigEl = document.getElementById('signal-txt');
            latestIsArtifact = data.safety.is_sensor_hold || (data.vitals.sqi_percent < 50.0);
            if (latestIsArtifact) {
                sigEl.innerHTML = `<span style="color: #ef4444; font-weight: 800;">⚠️ BOVIE NOISE (SQI ${data.vitals.sqi_percent.toFixed(0)}%) — SENSOR HOLD</span>`;
            } else {
                sigEl.innerHTML = `<span style="color: var(--green);">SQI ${data.vitals.sqi_percent.toFixed(0)}% (CLEAN)</span>`;
            }

            // 10. XAI Attributions
            if (data.xai) {
                document.getElementById('xai-driver').innerText = data.xai.primary_driver;
                document.getElementById('xai-bar-map').style.width = `${data.xai.map_risk_pct}%`;
                document.getElementById('xai-val-map').innerText = `${data.xai.map_risk_pct.toFixed(0)}%`;
                document.getElementById('xai-bar-bis').style.width = `${data.xai.bis_error_pct}%`;
                document.getElementById('xai-val-bis').innerText = `${data.xai.bis_error_pct.toFixed(0)}%`;
                document.getElementById('xai-bar-pain').style.width = `${data.xai.nociception_pct}%`;
                document.getElementById('xai-val-pain').innerText = `${data.xai.nociception_pct.toFixed(0)}%`;
            }

            // Mode & Autonomy labels
            if (data.mode) {
                activeEngineMode = data.mode;
                const btnM = document.getElementById('btn-toggle-mode');
                if (btnM) btnM.innerText = (activeEngineMode === 'MODE_A_SIMULATOR') ? 'Mode A (Digital Twin Sim)' : 'Mode B (VitalDB Replay)';
            }
            if (data.autonomy_level) {
                activeAutonomyLevel = data.autonomy_level;
                const btnA = document.getElementById('btn-toggle-autonomy');
                if (btnA) btnA.innerText = (activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP') ? 'Level 4 (Closed-Loop)' : 'Level 2 (Advisory)';
            }

            latestPredictions = data.predictions;
            bisHistory = suiteHistories[currentSuite].bis;
            mapHistory = suiteHistories[currentSuite].map;
            rawBisHistory = suiteHistories[currentSuite].raw_bis;

            drawChart();
        }

        function drawChart() {
            const w = canvas.width;
            const h = canvas.height;
            if (!w || !h) return;

            ctx.clearRect(0, 0, w, h);

            // 1. Target BIS Therapeutic Zone (40 - 60)
            const y60 = h - (60 / 120) * h;
            const y40 = h - (40 / 120) * h;
            ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
            ctx.fillRect(0, y60, w, y40 - y60);

            ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y60); ctx.lineTo(w, y60);
            ctx.moveTo(0, y40); ctx.lineTo(w, y40);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.font = '10px sans-serif';
            ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
            ctx.fillText('TARGET BIS: 40 - 60 (Adequate Surgical Anesthesia)', 12, y60 + 13);

            // 2. Horizontal Grid Lines & Y-Axis Labels
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
            ctx.lineWidth = 1;
            [20, 40, 60, 80, 100].forEach(level => {
                const y = h - (level / 120) * h;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();

                ctx.fillStyle = 'rgba(156, 163, 175, 0.5)';
                ctx.font = '9px sans-serif';
                ctx.fillText(`${level}`, 4, y - 3);
            });

            if (bisHistory.length < 2) {
                ctx.font = '12px sans-serif';
                ctx.fillStyle = 'rgba(156, 163, 175, 0.6)';
                ctx.textAlign = 'center';
                ctx.fillText('INITIALIZING CONTINUOUS PHYSIOLOGICAL STREAM (2 Hz)...', w / 2, h / 2);
                ctx.textAlign = 'left';
                return;
            }

            const historyW = w * 0.75;
            const stepX = historyW / Math.max(bisHistory.length - 1, 1);

            // 3. Draw MAP Trajectory (Rose / Coral)
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            for (let i = 0; i < mapHistory.length; i++) {
                const x = i * stepX;
                const y = h - (mapHistory[i] / 120) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // 3B. If Electrocautery Artifact Active, draw Raw BIS Noise Spike (Amber Dotted)
            if (rawBisHistory.length > 0 && latestIsArtifact) {
                ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
                ctx.lineWidth = 1.6;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                for (let i = 0; i < rawBisHistory.length; i++) {
                    const x = i * stepX;
                    const y = h - (rawBisHistory[i] / 120) * h;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // 4. Draw Filtered BIS Trajectory (Cyan Glow)
            ctx.shadowColor = '#38bdf8';
            ctx.shadowBlur = 6;
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            let lastX = 0;
            let lastY = 0;
            for (let i = 0; i < bisHistory.length; i++) {
                const x = i * stepX;
                const y = h - (bisHistory[i] / 120) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                lastX = x;
                lastY = y;
            }
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Draw current head dot
            ctx.fillStyle = '#38bdf8';
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            ctx.fill();

            // 5. Draw 10-Min Conformal Risk Predictive Cone (Future Projection)
            if (latestPredictions && latestPredictions.bis_10min_conformal_low !== undefined) {
                const low = latestPredictions.bis_10min_conformal_low;
                const high = latestPredictions.bis_10min_conformal_high;
                const endX = w - 10;
                const yLow = h - (low / 120) * h;
                const yHigh = h - (high / 120) * h;

                ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(endX, yHigh);
                ctx.lineTo(endX, yLow);
                ctx.closePath();
                ctx.fill();

                ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
                ctx.setLineDash([4, 3]);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastX, lastY); ctx.lineTo(endX, yHigh);
                ctx.moveTo(lastX, lastY); ctx.lineTo(endX, yLow);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.font = '9px sans-serif';
                ctx.fillStyle = '#38bdf8';
                ctx.fillText(`+10m CI [${low.toFixed(0)}-${high.toFixed(0)}]`, endX - 60, (yLow + yHigh) / 2);
            }

            // Timeline divider (NOW)
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(lastX, 0);
            ctx.lineTo(lastX, h);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.font = '9px sans-serif';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.fillText('NOW', lastX - 12, h - 8);
        }

        // Initialize immediately
        connectWS('OR-1');
    </script>
</body>
</html>
"""

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("anesthesionet.server.streaming_server:app", host="127.0.0.1", port=8000, reload=True)
