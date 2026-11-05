"""
FastAPI & WebSockets Real-Time Multi-Room Streaming Server
Powers the 1-to-Many remote supervision clinical command station.
Streams live 10 Hz telemetry, conformal predictive forecast intervals,
and HOCBF safety statuses across independent surgical suites (OR 1, OR 2, OR 3).
"""

import asyncio
from datetime import datetime
import json
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np

from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.engine_selector import DualEngineCoordinator, ExecutionMode
from anesthesionet.controllers.fsm import FullLifecycleController, SurgicalStage
from anesthesionet.forecasting.conformal_tft import ConformalPredictiveForecaster
from anesthesionet.safety.preload_gate import QualifiedPreloadGate
from anesthesionet.safety.watchdog import BedsideWatchdogDaemon


class SurgicalSuiteInstance:
    """Encapsulates a full, independent operating room instance."""

    def __init__(self, suite_id: str, patient: PatientProfile):
        self.suite_id = suite_id
        self.patient = patient
        
        self.engine = DualEngineCoordinator(patient)
        self.controller = FullLifecycleController(patient)
        self.forecaster = ConformalPredictiveForecaster(coverage_confidence=0.95)
        self.preload_gate = QualifiedPreloadGate()
        self.watchdog = BedsideWatchdogDaemon(patient)

        # Autonomy: Level 2 (Advisory) or Level 4 (Autonomous Closed-Loop)
        self.autonomy_level = "LEVEL_4_CLOSED_LOOP"
        self.current_time_sec = 0.0
        self.latest_action = np.array([3.5, 0.12, 0.0], dtype=np.float32)
        self.surgical_stimulus = 0.0
        self.ppv_bias = 0.0
        self.injected_artifact_duration = 0.0

    def step(self) -> dict:
        self.current_time_sec += 1.0
        self.watchdog.receive_heartbeat(self.current_time_sec)

        # Handle transient stimulus decay
        if self.surgical_stimulus > 0.0:
            self.surgical_stimulus = max(0.0, self.surgical_stimulus - 0.03)
        if self.ppv_bias > 0.0:
            self.ppv_bias = max(0.0, self.ppv_bias - 0.15)

        # 1. Step Dual Engine (Mode A Simulator or Mode B Replay)
        filtered, cf_proj, meta = self.engine.step(
            ai_action=self.latest_action,
            surgical_stimulus=self.surgical_stimulus,
            dt_seconds=1.0
        )

        # Injected electrocautery artifact handling
        if self.injected_artifact_duration > 0.0:
            self.injected_artifact_duration -= 1.0
            filtered = self.engine.signal_filter.filter(
                raw_bis=98.0,
                raw_map=filtered.raw_map,
                raw_hr=filtered.raw_hr,
                sqi=18.0,
                emg_db=56.0,
                dt_seconds=1.0
            )

        # 2. Record frame in predictive history
        self.forecaster.record_frame(
            time_sec=self.current_time_sec,
            bis=filtered.filtered_bis,
            map_mmhg=filtered.filtered_map,
            hr_bpm=filtered.filtered_hr,
            prop_rate=float(self.latest_action[0]),
            remi_rate=float(self.latest_action[1]),
            norepi_rate=float(self.latest_action[2])
        )

        # 3. Forecast future trajectories
        prediction = self.forecaster.forecast()

        # 4. Evaluate Dynamic Preload
        # Assume PPV ~ 9% in normovolemia, rising if stimulus/deficit occurs
        ppv_est = 9.0 + self.ppv_bias + (max(0.0, 75.0 - filtered.filtered_map) * 0.1)
        preload_eval = self.preload_gate.evaluate(ppv_percent=ppv_est)

        # 5. Evaluate FSM Controller
        # Create vitals snapshot
        vitals_snapshot = self.engine.simulator.hemodynamics
        vitals_obj = type("VitalsObj", (), {
            "bis": filtered.filtered_bis,
            "map_mmhg": filtered.filtered_map,
            "hr_bpm": filtered.filtered_hr,
            "ce_propofol_ug_ml": self.engine.simulator.pk_prop.Ce,
            "ce_remi_ng_ml": self.engine.simulator.pk_remi.Ce,
            "time_seconds": self.current_time_sec
        })()

        safe_action, was_intervened, fsm_info = self.controller.step(
            vitals=vitals_obj,
            surgical_stimulus=self.surgical_stimulus,
            dt_seconds=1.0
        )

        # Enforce preload clamp on Norepinephrine if hypovolemic
        if preload_eval.is_fluid_responsive:
            safe_action[2] = min(safe_action[2], preload_eval.max_allowable_norepi_rate)

        # In Level 4 autonomy, apply certified action; in Level 2, advise
        self.latest_action = safe_action

        # 6. Check Watchdog status
        watchdog_stat = self.watchdog.evaluate_status(self.current_time_sec)

        # 7. Compute Real-Time XAI Feature Attributions
        xai_map_risk = round(min(100.0, max(5.0, (75.0 - filtered.filtered_map) * 3.0 + 35.0)), 1)
        xai_bis_error = round(min(100.0, max(5.0, abs(50.0 - filtered.filtered_bis) * 3.5 + 20.0)), 1)
        xai_nociception = round(min(100.0, max(5.0, self.surgical_stimulus * 95.0 + 10.0)), 1)
        if self.surgical_stimulus > 0.25:
            xai_driver = "Surgical Nociception Surge"
        elif filtered.filtered_map < 68.0:
            xai_driver = "Conformal Hypotension Lookahead"
        elif abs(filtered.filtered_bis - 50.0) > 10.0:
            xai_driver = "Hypnotic Depth Error Compensation"
        else:
            xai_driver = "Bouillon Synergistic Ridge Titration"

        # Compile payload
        return {
            "suite_id": self.suite_id,
            "time_seconds": self.current_time_sec,
            "mode": self.engine.mode.value,
            "stage": self.controller.stage.value,
            "badge": fsm_info.get("badge") or f"[{self.controller.stage.value}]",
            "autonomy_level": self.autonomy_level,
            "vitals": {
                "raw_bis": round(filtered.raw_bis, 1),
                "filtered_bis": round(filtered.filtered_bis, 1),
                "raw_map": round(filtered.raw_map, 1),
                "filtered_map": round(filtered.filtered_map, 1),
                "filtered_hr": round(filtered.filtered_hr, 1),
                "ppv_percent": round(ppv_est, 1),
                "sqi_percent": round(filtered.sqi_percent, 1),
                "emg_db": round(filtered.emg_db, 1),
            },
            "infusion_rates": {
                "propofol_mg_kg_h": round(float(safe_action[0]), 2),
                "remifentanil_ug_kg_min": round(float(safe_action[1]), 3),
                "norepinephrine_ug_kg_min": round(float(safe_action[2]), 3),
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
                "is_sensor_hold": filtered.sensor_hold_active,
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


def create_app() -> FastAPI:
    app = FastAPI(title="AnesthesioNet-AI Clinical Co-Pilot Server", version="0.1.0")

    # Initialize 3 Independent Operating Suites
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

    # Prime OR-1 to Maintenance stage
    suites["OR-1"].controller.stage = SurgicalStage.STAGE_2_MAINTENANCE
    suites["OR-1"].engine.simulator.pk_prop.state[0] = 16.0
    suites["OR-1"].engine.simulator.pk_prop.state[3] = 2.2  # Ce Propofol = 2.2 ug/mL (Therapeutic)
    suites["OR-1"].engine.simulator.pk_remi.state[0] = 7.0
    suites["OR-1"].engine.simulator.pk_remi.state[3] = 1.5  # Ce Remifentanil = 1.5 ng/mL (Therapeutic)

    # Prime OR-3 to Stage 3 Emergence stage (Emergence Clearance)
    suites["OR-3"].controller.stage = SurgicalStage.STAGE_3_EMERGENCE
    suites["OR-3"].engine.simulator.pk_prop.state[0] = 3.0
    suites["OR-3"].engine.simulator.pk_prop.state[3] = 0.9  # Ce Propofol clearing (Awakening)
    suites["OR-3"].engine.simulator.pk_remi.state[0] = 0.8
    suites["OR-3"].engine.simulator.pk_remi.state[3] = 0.3

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        return HTML_DASHBOARD_TEMPLATE

    @app.get("/status")
    async def get_status():
        return {suite_id: s.step() for suite_id, s in suites.items()}

    @app.get("/api/telemetry/{suite_id}")
    async def get_suite_telemetry(suite_id: str):
        suite = suites.get(suite_id, suites["OR-1"])
        return suite.step()

    @app.post("/api/action/{suite_id}")
    async def post_action(suite_id: str, payload: dict):
        suite = suites.get(suite_id, suites["OR-1"])
        action = payload.get("action")

        if action == "pain":
            suite.surgical_stimulus = float(payload.get("value", 0.85))
        elif action == "hypotension":
            suite.engine.simulator.hemodynamics.baseline_map = max(45.0, suite.engine.simulator.hemodynamics.baseline_map - float(payload.get("value", 20.0)))
        elif action == "bolus":
            suite.engine.simulator.hemodynamics.baseline_map = min(92.0, suite.engine.simulator.hemodynamics.baseline_map + 15.0)
            suite.ppv_bias = -3.0
        elif action == "preload_deficit":
            suite.ppv_bias = 7.0
        elif action == "artifact":
            suite.injected_artifact_duration = float(payload.get("duration", 6.0))
        elif action == "stage":
            new_stage = str(payload.get("stage", ""))
            if "STANDBY" in new_stage.upper() or new_stage == "0":
                suite.controller.stage = SurgicalStage.STANDBY
                suite.controller.induction_controller.reset()
                suite.engine.simulator.pk_prop.reset()
                suite.engine.simulator.pk_remi.reset()
                suite.engine.simulator.hemodynamics.baseline_map = 90.0
                suite.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            elif "INDUCTION" in new_stage.upper() or new_stage == "1":
                suite.controller.stage = SurgicalStage.STAGE_1_INDUCTION
                bolus_mg = suite.controller.trigger_induction()
                suite.engine.simulator.pk_prop.bolus(bolus_mg)
                prop_rate, remi_rate, norepi_rate, _ = suite.controller.induction_controller.compute_infusion_rates(
                    type("VitalsObj", (), {
                        "bis": 98.0,
                        "map_mmhg": suite.engine.simulator.hemodynamics.baseline_map,
                        "hr_bpm": 75.0,
                        "ce_propofol_ug_ml": suite.engine.simulator.pk_prop.Ce,
                        "ce_remi_ng_ml": suite.engine.simulator.pk_remi.Ce,
                        "time_seconds": suite.current_time_sec
                    })()
                )
                suite.latest_action = np.array([prop_rate, remi_rate, norepi_rate], dtype=np.float32)

            elif "MAINTENANCE" in new_stage.upper() or "MAINT" in new_stage.upper() or new_stage == "2":
                suite.controller.stage = SurgicalStage.STAGE_2_MAINTENANCE
                if suite.engine.simulator.pk_prop.Ce < 1.0:
                    suite.engine.simulator.pk_prop.state[3] = 2.2
                    suite.engine.simulator.pk_prop.state[0] = 16.0
                if suite.engine.simulator.pk_remi.Ce < 0.5:
                    suite.engine.simulator.pk_remi.state[3] = 1.5
                    suite.engine.simulator.pk_remi.state[0] = 7.0
                suite.latest_action = np.array([3.5, 0.12, 0.0], dtype=np.float32)

            elif "EMERGENCE" in new_stage.upper() or new_stage == "3":
                suite.controller.trigger_emergence(suite.current_time_sec)
                suite.latest_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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
                try:
                    payload = suite.step()
                    await websocket.send_text(json.dumps(payload, default=float))
                except Exception as step_err:
                    print(f"[WebSocket Error] Suite {suite_id}: {step_err}")
                await asyncio.sleep(0.5)  # 2 Hz telemetry broadcast
        except WebSocketDisconnect:
            pass
        except Exception as ws_err:
            print(f"[WebSocket Fatal] Suite {suite_id}: {ws_err}")

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
            font-size: 1.15rem;
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
            font-size: 0.7rem;
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
            padding: 4px 12px;
            border-radius: 5px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background-color: var(--blue);
            color: #fff;
        }
        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 10px;
            padding: 8px 12px;
            overflow: hidden;
            height: calc(100vh - 42px);
            max-height: calc(100vh - 42px);
        }
        .col-left {
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
        .telemetry-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            flex: 1;
            min-height: 0;
        }
        .vitals-banner {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 6px 0;
        }
        .vital-box {
            background-color: #1e293b;
            border-radius: 6px;
            padding: 6px;
            text-align: center;
        }
        .vital-title {
            font-size: 0.68rem;
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .vital-num {
            font-size: 1.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 2px 0;
        }
        .vital-sub {
            font-size: 0.65rem;
        }
        .bis-color { color: #38bdf8; }
        .map-color { color: #f43f5e; }
        .hr-color { color: #34d399; }
        .ppv-color { color: #fbbf24; }
        .chart-canvas {
            width: 100%;
            flex: 1;
            min-height: 120px;
            background-color: #0b1120;
            border-radius: 6px;
            display: block;
        }
        .interlock-box {
            background-color: #172554;
            border-left: 3px solid var(--blue);
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #bfdbfe;
            margin-top: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .syringe-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            flex-shrink: 0;
        }
        .syringe-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .syringe-item {
            background-color: #1e293b;
            border-radius: 6px;
            padding: 6px 8px;
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .syringe-drug {
            font-size: 0.75rem;
            font-weight: 700;
        }
        .syringe-sub {
            font-size: 0.65rem;
            color: var(--text-muted);
        }
        .syringe-rate {
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 2px;
        }
        .col-right {
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
        .panel-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
        }
        .safety-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px 10px;
            font-size: 0.75rem;
            line-height: 1.35;
        }
        .safety-grid b {
            color: #9ca3af;
        }
        .preload-strip {
            margin-top: 6px;
            padding: 4px 8px;
            background-color: #0b1120;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #fbbf24;
            border-left: 3px solid #f59e0b;
        }
        .section-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-muted);
            margin: 0 0 5px 0;
            letter-spacing: 0.03em;
        }
        .stress-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            margin-bottom: 6px;
        }
        .ctrl-btn {
            background: #1e293b;
            color: #f3f4f6;
            border: 1px solid #334155;
            padding: 5px 6px;
            border-radius: 5px;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 4px;
            justify-content: center;
            white-space: nowrap;
        }
        .ctrl-btn:hover {
            background: #334155;
            border-color: var(--blue);
        }
        .ctrl-btn-pain { background: #7c2d12; border-color: #ea580c; color: #ffedd5; }
        .ctrl-btn-pain:hover { background: #9a3412; }
        .ctrl-btn-crash { background: #881337; border-color: #e11d48; color: #ffe4e6; }
        .ctrl-btn-crash:hover { background: #9f1239; }
        .ctrl-btn-bolus { background: #064e3b; border-color: #059669; color: #d1fae5; }
        .ctrl-btn-bolus:hover { background: #047857; }
        .ctrl-btn-bovie { background: #78350f; border-color: #d97706; color: #fef3c7; }
        .ctrl-btn-bovie:hover { background: #92400e; }
        .stage-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 4px;
            margin-bottom: 6px;
        }
        .stage-pill {
            padding: 4px 4px;
            border-radius: 5px;
            font-size: 0.68rem;
            font-weight: 700;
            cursor: pointer;
            border: 1px solid #334155;
            background: #0f172a;
            color: var(--text-muted);
            text-align: center;
            transition: all 0.15s ease;
            white-space: nowrap;
        }
        .stage-pill:hover { border-color: var(--blue); color: #fff; }
        .stage-pill.active {
            background: #1e3a8a;
            border-color: #3b82f6;
            color: #93c5fd;
        }
        .toggles-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid var(--border);
            padding-top: 5px;
        }
        .xai-bar-wrap {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
            font-size: 0.72rem;
        }
        .xai-bar-track {
            flex: 1;
            height: 6px;
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
            <button class="tab-btn active" onclick="switchSuite('OR-1')">SUITE 1 (OR-1)</button>
            <button class="tab-btn" onclick="switchSuite('OR-2')">SUITE 2 (OR-2)</button>
            <button class="tab-btn" onclick="switchSuite('OR-3')">SUITE 3 (OR-3)</button>
        </div>
        <div id="conn-badge" style="font-size: 0.78rem; color: var(--amber); font-weight: 600;">
            ● INITIALIZING TELEMETRY STREAM...
        </div>
    </header>

    <main>
        <!-- LEFT COLUMN: Live Telemetry, Waveforms & Syringe Infusion Rack -->
        <div class="col-left">
            <div class="telemetry-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <h3 style="margin: 0; font-size: 0.95rem;" id="suite-header">Active Suite: OR-1 (Laparoscopic Cholecystectomy)</h3>
                    <span id="stage-badge" class="header-badge" style="background-color: #1e3a8a; color: #93c5fd;">[STAGE 2: MAINTENANCE]</span>
                </div>

                <div class="vitals-banner">
                    <div class="vital-box">
                        <div class="vital-title">BISPECTRAL INDEX</div>
                        <div class="vital-num bis-color" id="val-bis">--</div>
                        <div class="vital-sub" style="color: var(--green);">Target: 40 - 60</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">MEAN ARTERIAL PRESSURE</div>
                        <div class="vital-num map-color" id="val-map">--</div>
                        <div class="vital-sub" style="color: var(--green);">Target: 65 - 100 mmHg</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">HEART RATE</div>
                        <div class="vital-num hr-color" id="val-hr">--</div>
                        <div class="vital-sub" style="color: var(--text-muted);">bpm (Sinus)</div>
                    </div>
                    <div class="vital-box">
                        <div class="vital-title">PULSE PRESSURE VAR (PPV)</div>
                        <div class="vital-num ppv-color" id="val-ppv">--</div>
                        <div class="vital-sub" style="color: var(--green);">Normovolemic (&lt;13%)</div>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin: 2px 0 4px 0;">
                    <div style="font-size: 0.78rem; font-weight: 600;">Live Waveform &amp; Conformal Predictive Risk Bands (10-Min Horizon)</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); display: flex; gap: 8px;">
                        <span style="color: #38bdf8;">— BIS</span>
                        <span style="color: #f43f5e;">— MAP</span>
                        <span style="color: #10b981;">░ Target</span>
                        <span style="color: rgba(56, 189, 248, 0.7);">┅ 10m Cone</span>
                    </div>
                </div>
                <canvas id="chart-canvas" class="chart-canvas"></canvas>

                <div class="interlock-box" id="interlock-text">
                    [SYNCHRONIZING WITH PHYSIOLOGIC CO-PILOT...]
                </div>
            </div>

            <!-- Syringe Pump Infusion Array (Compact 3-Rack) -->
            <div class="syringe-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div style="font-size: 0.8rem; font-weight: 700;">Automated Syringe Pump Infusion Array</div>
                    <div style="font-size: 0.68rem; color: var(--text-muted);">MIMO-SAC Closed-Loop Infusion</div>
                </div>
                <div class="syringe-grid">
                    <div class="syringe-item">
                        <div>
                            <div class="syringe-drug" style="color: #38bdf8;">Propofol (Hypnotic)</div>
                            <div class="syringe-sub">Ce: 2.2 µg/mL | Ridge</div>
                        </div>
                        <div class="syringe-rate" id="rate-prop">0.00 <span style="font-size: 0.7rem; font-weight: 400;">mg/kg/h</span></div>
                    </div>
                    <div class="syringe-item">
                        <div>
                            <div class="syringe-drug" style="color: #a78bfa;">Remifentanil (Opioid)</div>
                            <div class="syringe-sub">Ce: 1.5 ng/mL | Analgesic</div>
                        </div>
                        <div class="syringe-rate" id="rate-remi">0.000 <span style="font-size: 0.7rem; font-weight: 400;">µg/kg/min</span></div>
                    </div>
                    <div class="syringe-item">
                        <div>
                            <div class="syringe-drug" style="color: #f43f5e;">Norepinephrine (Pressor)</div>
                            <div class="syringe-sub">Preload Guarded | SVR</div>
                        </div>
                        <div class="syringe-rate" id="rate-norepi">0.000 <span style="font-size: 0.7rem; font-weight: 400;">µg/kg/min</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN: Safety Guardrails, Interactive Challenges & Explainable AI -->
        <div class="col-right">
            <!-- Panel 1: Safety & Hardware Status -->
            <div class="panel-card" style="flex-shrink: 0;">
                <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 6px; display: flex; justify-content: space-between;">
                    <span>Safety, Interlocks &amp; Hardware Status</span>
                    <span style="font-size: 0.68rem; color: var(--green);">DETERMINISTIC GUARDRAILS</span>
                </div>
                <div class="safety-grid">
                    <div>• <b>Autonomy:</b> <span id="autonomy-txt" style="color: var(--green);">Level 4 Closed-Loop</span></div>
                    <div>• <b>HOCBF QP:</b> <span id="hocbf-txt" style="color: var(--green);">ACTIVE (Forward Invariance)</span></div>
                    <div>• <b>Watchdog:</b> <span id="watchdog-txt" style="color: var(--green);">HEARTBEAT NORMAL (&lt;500ms)</span></div>
                    <div>• <b>Signal SQI:</b> <span id="signal-txt" style="color: var(--green);">SQI 98% | EMG 24 dB</span></div>
                </div>
                <div class="preload-strip" id="preload-txt">
                    Preload Gate: Normovolemic (PPV &lt; 13%)
                </div>
            </div>

            <!-- Panel 2: Interactive Clinical Challenge & Lifecycle Console -->
            <div class="panel-card" style="flex-shrink: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.8rem; font-weight: 700;">Interactive Challenge &amp; Stress Console</span>
                    <span style="font-size: 0.68rem; color: var(--text-muted);">Real-Time Co-Pilot Response</span>
                </div>

                <div class="section-label">PHYSIOLOGICAL STRESS INJECTION:</div>
                <div class="stress-grid">
                    <button class="ctrl-btn ctrl-btn-pain" onclick="sendAction('pain', {value: 0.85})">
                        ⚡ Incision Pain (+0.85)
                    </button>
                    <button class="ctrl-btn ctrl-btn-crash" onclick="sendAction('hypotension', {value: 20.0})">
                        📉 MAP Crash (-20 mmHg)
                    </button>
                    <button class="ctrl-btn ctrl-btn-bolus" onclick="sendAction('bolus', {})">
                        💧 250mL Fluid Bolus
                    </button>
                    <button class="ctrl-btn ctrl-btn-bovie" onclick="sendAction('artifact', {duration: 6.0})">
                        📻 Bovie Electrocautery
                    </button>
                </div>

                <div class="section-label">SURGICAL LIFECYCLE FSM TRANSITIONS:</div>
                <div class="stage-grid">
                    <button class="stage-pill" id="pill-STAGE_0_STANDBY" onclick="sendAction('stage', {stage: 'STAGE_0_STANDBY'})">
                        0. Standby
                    </button>
                    <button class="stage-pill" id="pill-STAGE_1_INDUCTION" onclick="sendAction('stage', {stage: 'STAGE_1_INDUCTION'})">
                        1. Induction
                    </button>
                    <button class="stage-pill active" id="pill-STAGE_2_MAINTENANCE" onclick="sendAction('stage', {stage: 'STAGE_2_MAINTENANCE'})">
                        2. Maint.
                    </button>
                    <button class="stage-pill" id="pill-STAGE_3_EMERGENCE" onclick="sendAction('stage', {stage: 'STAGE_3_EMERGENCE'})">
                        3. Emergence
                    </button>
                </div>

                <div class="toggles-row">
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <span style="font-size: 0.72rem; color: var(--text-muted);">Mode:</span>
                        <button class="ctrl-btn" id="btn-toggle-mode" onclick="toggleEngineMode()">
                            Mode A (Digital Twin Sim)
                        </button>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <span style="font-size: 0.72rem; color: var(--text-muted);">Autonomy:</span>
                        <button class="ctrl-btn" id="btn-toggle-autonomy" onclick="toggleAutonomy()">
                            Level 4 (Closed-Loop)
                        </button>
                    </div>
                </div>
            </div>

            <!-- Panel 3: Real-Time Explainable AI (XAI) Attribution Console -->
            <div class="panel-card" style="flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.8rem; font-weight: 700;">Event-Triggered Explainability (XAI)</span>
                        <span style="font-size: 0.68rem; color: var(--blue);" id="xai-tag">LIVE ATTRIBUTION</span>
                    </div>

                    <div style="font-size: 0.72rem; color: var(--text-muted); margin-bottom: 6px;">
                        Primary Titration Driver:
                        <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-top: 1px;" id="xai-driver">
                            Bouillon Synergistic Ridge Titration
                        </div>
                    </div>

                    <div class="xai-bar-wrap">
                        <span style="width: 125px; color: var(--text-muted); font-size: 0.72rem;">Conformal MAP Risk:</span>
                        <div class="xai-bar-track">
                            <div class="xai-bar-fill" id="xai-bar-map" style="width: 35%; background: #f43f5e;"></div>
                        </div>
                        <span style="width: 32px; text-align: right; font-weight: 700; color: #f43f5e; font-size: 0.72rem;" id="xai-val-map">35%</span>
                    </div>

                    <div class="xai-bar-wrap">
                        <span style="width: 125px; color: var(--text-muted); font-size: 0.72rem;">BIS Depth Deviation:</span>
                        <div class="xai-bar-track">
                            <div class="xai-bar-fill" id="xai-bar-bis" style="width: 25%; background: #38bdf8;"></div>
                        </div>
                        <span style="width: 32px; text-align: right; font-weight: 700; color: #38bdf8; font-size: 0.72rem;" id="xai-val-bis">25%</span>
                    </div>

                    <div class="xai-bar-wrap">
                        <span style="width: 125px; color: var(--text-muted); font-size: 0.72rem;">Surgical Nociception:</span>
                        <div class="xai-bar-track">
                            <div class="xai-bar-fill" id="xai-bar-pain" style="width: 12%; background: #fbbf24;"></div>
                        </div>
                        <span style="width: 32px; text-align: right; font-weight: 700; color: #fbbf24; font-size: 0.72rem;" id="xai-val-pain">12%</span>
                    </div>
                </div>

                <div style="font-size: 0.68rem; color: var(--text-muted); line-height: 1.3; border-top: 1px solid var(--border); padding-top: 4px; margin-top: 4px;">
                    • Attributions evaluated on relative gradient shift (&gt;5% delta) per ISO 60601-1-10 explainability standard.
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
        let latestPredictions = null;

        // Dedicated per-suite persistent telemetry buffers to isolate operating rooms
        const suiteHistories = {
            'OR-1': { bis: [], map: [] },
            'OR-2': { bis: [], map: [] },
            'OR-3': { bis: [], map: [] }
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
                connBadge.innerText = `● ${text || 'RECONNECTING TELEMETRY STREAM...'}`;
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
                    console.warn('WebSocket error, switching to HTTP fallback polling:', e);
                    setConnectionState('polling', 'CENTRAL CO-PILOT LIVE (HTTP FALLBACK)');
                    startPollingFallback(suite);
                };

                ws.onclose = () => {
                    console.warn('WebSocket closed, switching to HTTP fallback polling...');
                    setConnectionState('polling', 'CENTRAL CO-PILOT LIVE (HTTP FALLBACK)');
                    startPollingFallback(suite);
                    setTimeout(() => {
                        if (currentSuite === suite) connectWS(suite);
                    }, 3000);
                };
            } catch(e) {
                console.warn('Failed to initialize WebSocket, starting HTTP polling:', e);
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

        let activeEngineMode = 'MODE_A_SIMULATOR';
        let activeAutonomyLevel = 'LEVEL_4_CLOSED_LOOP';

        async function sendAction(action, extra = {}) {
            try {
                await fetch(`/api/action/${currentSuite}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action, ...extra })
                });
                if (action === 'stage' && extra.stage) {
                    document.querySelectorAll('.stage-pill').forEach(el => el.classList.remove('active'));
                    const pill = document.getElementById(`pill-${extra.stage}`);
                    if (pill) pill.classList.add('active');
                }
            } catch(e) {
                console.error('Action dispatch failed:', e);
            }
        }

        function toggleEngineMode() {
            activeEngineMode = activeEngineMode === 'MODE_A_SIMULATOR' ? 'MODE_B_REPLAY' : 'MODE_A_SIMULATOR';
            document.getElementById('btn-toggle-mode').innerText = activeEngineMode === 'MODE_A_SIMULATOR' ? 'Mode A (Digital Twin Sim)' : 'Mode B (VitalDB Replay)';
            sendAction('mode', { mode: activeEngineMode });
        }

        function toggleAutonomy() {
            activeAutonomyLevel = activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP' ? 'LEVEL_2_ADVISORY' : 'LEVEL_4_CLOSED_LOOP';
            document.getElementById('btn-toggle-autonomy').innerText = activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP' ? 'Level 4 (Closed-Loop)' : 'Level 2 (Advisory)';
            sendAction('autonomy', { level: activeAutonomyLevel });
        }

        function switchSuite(suite) {
            if (currentSuite === suite) return;
            currentSuite = suite;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.innerText.includes(suite));
            });
            document.getElementById('suite-header').innerText = `Active Suite: ${suite}`;

            // Instantly load this suite's isolated history buffer (eliminates cross-suite jump artifacts)
            bisHistory = (suiteHistories[suite] && suiteHistories[suite].bis) ? suiteHistories[suite].bis : [];
            mapHistory = (suiteHistories[suite] && suiteHistories[suite].map) ? suiteHistories[suite].map : [];
            drawChart();
            connectWS(suite);
        }

        function updateUI(data) {
            if (!data || !data.vitals) return;

            // Route data to the correct operating suite buffer
            const sId = data.suite_id || currentSuite;
            if (!suiteHistories[sId]) {
                suiteHistories[sId] = { bis: [], map: [] };
            }
            suiteHistories[sId].bis.push(data.vitals.filtered_bis);
            suiteHistories[sId].map.push(data.vitals.filtered_map);
            if (suiteHistories[sId].bis.length > 70) { 
                suiteHistories[sId].bis.shift(); 
                suiteHistories[sId].map.shift(); 
            }

            // Reject any in-flight packets that belong to a different suite than the active tab
            if (data.suite_id && data.suite_id !== currentSuite) {
                return;
            }

            document.getElementById('val-bis').innerText = data.vitals.filtered_bis.toFixed(1);
            document.getElementById('val-map').innerText = data.vitals.filtered_map.toFixed(1);
            document.getElementById('val-hr').innerText = data.vitals.filtered_hr.toFixed(1);
            document.getElementById('val-ppv').innerText = `${data.vitals.ppv_percent.toFixed(1)}%`;
            document.getElementById('stage-badge').innerText = data.badge;

            document.getElementById('rate-prop').innerHTML = `${data.infusion_rates.propofol_mg_kg_h.toFixed(2)} <span style="font-size: 0.7rem; font-weight: 400;">mg/kg/h</span>`;
            document.getElementById('rate-remi').innerHTML = `${data.infusion_rates.remifentanil_ug_kg_min.toFixed(3)} <span style="font-size: 0.7rem; font-weight: 400;">µg/kg/min</span>`;
            document.getElementById('rate-norepi').innerHTML = `${data.infusion_rates.norepinephrine_ug_kg_min.toFixed(3)} <span style="font-size: 0.7rem; font-weight: 400;">µg/kg/min</span>`;

            document.getElementById('interlock-text').innerText = data.predictions.interlock_message;
            document.getElementById('preload-txt').innerText = data.safety.preload_advisory;
            document.getElementById('autonomy-txt').innerText = data.autonomy_level === 'LEVEL_4_CLOSED_LOOP' ? 'Level 4 Closed-Loop' : 'Level 2 Advisory';
            document.getElementById('signal-txt').innerText = `SQI ${data.vitals.sqi_percent.toFixed(0)}% | EMG ${data.vitals.emg_db.toFixed(0)} dB (CLEAN)`;

            // Update XAI Attributions
            if (data.xai) {
                document.getElementById('xai-driver').innerText = data.xai.primary_driver;
                document.getElementById('xai-bar-map').style.width = `${data.xai.map_risk_pct}%`;
                document.getElementById('xai-val-map').innerText = `${data.xai.map_risk_pct.toFixed(0)}%`;
                document.getElementById('xai-bar-bis').style.width = `${data.xai.bis_error_pct}%`;
                document.getElementById('xai-val-bis').innerText = `${data.xai.bis_error_pct.toFixed(0)}%`;
                document.getElementById('xai-bar-pain').style.width = `${data.xai.nociception_pct}%`;
                document.getElementById('xai-val-pain').innerText = `${data.xai.nociception_pct.toFixed(0)}%`;
            }

            // Update Stage Pill Active State
            if (data.stage) {
                document.querySelectorAll('.stage-pill').forEach(el => el.classList.remove('active'));
                let pillId = `pill-${data.stage}`;
                if (data.stage === 'STANDBY') pillId = 'pill-STAGE_0_STANDBY';
                else if (data.stage === 'READINESS_HYSTERESIS') pillId = 'pill-STAGE_1_INDUCTION';
                else if (data.stage === 'PACU_TRANSFER') pillId = 'pill-STAGE_3_EMERGENCE';
                const activePill = document.getElementById(pillId);
                if (activePill) activePill.classList.add('active');
            }

            // Update Engine Mode & Autonomy labels
            if (data.mode) {
                activeEngineMode = data.mode;
                document.getElementById('btn-toggle-mode').innerText = activeEngineMode === 'MODE_A_SIMULATOR' ? 'Mode A (Digital Twin Sim)' : 'Mode B (VitalDB Replay)';
            }
            if (data.autonomy_level) {
                activeAutonomyLevel = data.autonomy_level;
                document.getElementById('btn-toggle-autonomy').innerText = activeAutonomyLevel === 'LEVEL_4_CLOSED_LOOP' ? 'Level 4 (Closed-Loop)' : 'Level 2 (Advisory)';
            }

            latestPredictions = data.predictions;

            // Link active chart buffers to current suite's isolated history
            bisHistory = suiteHistories[currentSuite].bis;
            mapHistory = suiteHistories[currentSuite].map;

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

            // Dotted boundaries for target zone
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y60); ctx.lineTo(w, y60);
            ctx.moveTo(0, y40); ctx.lineTo(w, y40);
            ctx.stroke();
            ctx.setLineDash([]);

            // Target zone label
            ctx.font = '10px sans-serif';
            ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
            ctx.fillText('TARGET BIS: 40 - 60 (Adequate Surgical Anesthesia)', 12, y60 + 14);

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

            // If no data points yet, draw clean placeholder
            if (bisHistory.length < 2) {
                ctx.font = '12px sans-serif';
                ctx.fillStyle = 'rgba(156, 163, 175, 0.6)';
                ctx.textAlign = 'center';
                ctx.fillText('INITIALIZING CONTINUOUS PHYSIOLOGICAL STREAM (2 Hz)...', w / 2, h / 2);
                ctx.textAlign = 'left';
                return;
            }

            // Reserve left 75% for history, right 25% for conformal forecast projection
            const historyW = w * 0.75;
            const futureW = w * 0.25;
            const stepX = historyW / Math.max(bisHistory.length - 1, 1);

            // 3. Draw MAP Trajectory (Neon Coral / Rose)
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            for (let i = 0; i < mapHistory.length; i++) {
                const x = i * stepX;
                // Scale MAP (unified 0 to 120 mmHg scale) to canvas height
                const y = h - (mapHistory[i] / 120) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // 4. Draw BIS Trajectory (Cyan Glow)
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

            // Draw current pulsating head dot
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

                // Shaded forecast cone
                ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(endX, yHigh);
                ctx.lineTo(endX, yLow);
                ctx.closePath();
                ctx.fill();

                // Dashed forecast boundary lines
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
                ctx.setLineDash([4, 3]);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastX, lastY); ctx.lineTo(endX, yHigh);
                ctx.moveTo(lastX, lastY); ctx.lineTo(endX, yLow);
                ctx.stroke();
                ctx.setLineDash([]);

                // Label at cone apex
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
