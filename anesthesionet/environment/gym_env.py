"""
Gymnasium Anesthesia Closed-Loop Environment
Wraps the 3-compartment coupled PK/PD digital twin into a standard reinforcement learning
environment for multi-drug continuous infusion control.
"""

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.simulator import PatientSimulator, PatientVitals


class AnesthesiaClosedLoopEnv(gym.Env):
    """
    Continuous Multi-Input Multi-Output (MIMO) Gymnasium Environment.
    
    Actions:
        0: Propofol delivery rate (0.0 to 12.0 mg/kg/h)
        1: Remifentanil delivery rate (0.0 to 1.0 ug/kg/min)
        2: Norepinephrine delivery rate (0.0 to 0.2 ug/kg/min)
        
    Observations:
        0: BIS Index (0.0 to 100.0)
        1: Mean Arterial Pressure (20.0 to 180.0 mmHg)
        2: Heart Rate (30.0 to 180.0 bpm)
        3: Pulse Pressure Variation (4.0 to 30.0 %)
        4: Ce Propofol (0.0 to 15.0 ug/mL)
        5: Ce Remifentanil (0.0 to 20.0 ng/mL)
        6: Current Norepinephrine Ce (0.0 to 0.5 ug/kg/min)
        7: Active Surgical Noxious Stimulus (0.0 to 1.0)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        patient: PatientProfile | None = None,
        substep_dt: float = 0.1,
        control_interval_sec: float = 1.0,
        max_duration_sec: float = 3600.0,
    ):
        super().__init__()
        
        self.patient = patient or PatientProfile(
            age_years=45.0,
            weight_kg=70.0,
            height_cm=175.0,
            sex="male"
        )
        self.substep_dt = substep_dt
        self.control_interval_sec = control_interval_sec
        self.max_duration_sec = max_duration_sec
        
        self.simulator = PatientSimulator(self.patient)

        # Action bounds
        # [Propofol (mg/kg/h), Remifentanil (ug/kg/min), Norepinephrine (ug/kg/min)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([12.0, 1.0, 0.2], dtype=np.float32),
            dtype=np.float32
        )

        # Observation bounds
        # [BIS, MAP, HR, PPV, Ce_Prop, Ce_Remi, Ce_Norepi, Surgical_Stimulus]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 20.0, 30.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 180.0, 180.0, 30.0, 15.0, 20.0, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.last_action = np.zeros(3, dtype=np.float32)
        self.current_stimulus = 0.0
        self.fluid_deficit = 1.0
        self.current_vitals: PatientVitals | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        
        self.simulator.reset()
        self.last_action.fill(0.0)
        self.current_stimulus = 0.0
        self.fluid_deficit = 1.0

        # Step once with zero drug to get baseline
        self.current_vitals = self.simulator.step(
            propofol_rate_mg_kg_h=0.0,
            remifentanil_rate_ug_kg_min=0.0,
            norepinephrine_rate_ug_kg_min=0.0,
            dt_seconds=self.substep_dt
        )

        obs = self._get_obs()
        info = {"vitals": self.current_vitals, "time_seconds": 0.0}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        v = self.current_vitals
        return np.array([
            v.bis,
            v.map_mmhg,
            v.hr_bpm,
            v.ppv_percent,
            v.ce_propofol_ug_ml,
            v.ce_remi_ng_ml,
            self.simulator.hemodynamics.ce_norepi,
            self.current_stimulus
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Clip action to safety bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        prop_rate = float(action[0])
        remi_rate = float(action[1])
        norepi_rate = float(action[2])

        # Integrate internal physics substeps across control interval
        num_substeps = max(1, int(self.control_interval_sec / self.substep_dt))
        for _ in range(num_substeps):
            self.current_vitals = self.simulator.step(
                propofol_rate_mg_kg_h=prop_rate,
                remifentanil_rate_ug_kg_min=remi_rate,
                norepinephrine_rate_ug_kg_min=norepi_rate,
                surgical_stimulus=self.current_stimulus,
                fluid_deficit_factor=self.fluid_deficit,
                dt_seconds=self.substep_dt
            )

        reward = self._compute_reward(action)
        self.last_action = action.copy()

        # Termination: severe sustained cardiac collapse or max time
        terminated = bool(self.current_vitals.map_mmhg < 30.0 or self.current_vitals.hr_bpm < 30.0)
        truncated = bool(self.simulator.simulation_time_sec >= self.max_duration_sec)

        obs = self._get_obs()
        info = {
            "vitals": self.current_vitals,
            "time_seconds": self.simulator.simulation_time_sec,
            "terminated_reason": "circulatory_collapse" if terminated else None
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> float:
        v = self.current_vitals

        # 1. Hypnosis Target Reward (Target BIS = 50, zone 40 - 60)
        bis_error = abs(v.bis - 50.0)
        if 40.0 <= v.bis <= 60.0:
            r_bis = 1.0 - (bis_error / 10.0) * 0.2  # Positive reward inside target zone
        else:
            r_bis = -((bis_error / 10.0) ** 2)

        # 2. Hemodynamic Target (Target MAP = 75, safe zone 65 - 100)
        if 65.0 <= v.map_mmhg <= 100.0:
            r_map = 0.5
        elif v.map_mmhg < 65.0:
            # Steep penalty for hypotension
            r_map = -2.0 * (((65.0 - v.map_mmhg) / 5.0) ** 2)
        else:
            # Moderate penalty for hypertension
            r_map = -0.5 * (((v.map_mmhg - 100.0) / 10.0) ** 2)

        # 3. Action Smoothness Penalty (penalize abrupt jerky changes)
        delta_action = action - self.last_action
        r_smooth = -0.1 * float(np.sum(delta_action ** 2))

        # 4. Minto/Bouillon Interaction Surface Prior
        # Penalizes running high propofol without opioid or vice versa
        optimal_remi = self.simulator.pd_synergy.evaluate_optimal_remi_prior(v.ce_propofol_ug_ml)
        remi_error = abs(v.ce_remi_ng_ml - optimal_remi)
        r_synergy = -0.2 * remi_error

        return float(r_bis + r_map + r_smooth + r_synergy)

    def set_surgical_stimulus(self, intensity: float) -> None:
        """Sets external noxious surgical stimulus (0.0 = none, 1.0 = deep incision)."""
        self.current_stimulus = float(np.clip(intensity, 0.0, 1.0))

    def set_fluid_deficit(self, factor: float) -> None:
        """Sets intravascular volume status (1.0 = normovolemic, 1.5 = hypovolemic)."""
        self.fluid_deficit = max(0.5, float(factor))
