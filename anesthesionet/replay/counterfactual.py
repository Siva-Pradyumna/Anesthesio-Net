"""
Counterfactual Transfer Estimator & Doubly Robust Off-Policy Evaluator (DR-OPE)
Allows the AI Co-Pilot to run in "Shadow Mode" on historical VitalDB cases,
projecting what patient vitals would have been if the AI's recommendations had been administered.
"""

from dataclasses import dataclass
import numpy as np
from anesthesionet.physiology.patient import PatientProfile
from anesthesionet.physiology.simulator import PatientSimulator


@dataclass
class CounterfactualProjection:
    time_seconds: float
    dataset_bis: float
    counterfactual_bis: float
    dataset_map: float
    counterfactual_map: float
    doctor_action: np.ndarray        # [prop_mg_kg_h, remi_ug_kg_min, norepi_ug_kg_min]
    ai_action: np.ndarray            # [prop_mg_kg_h, remi_ug_kg_min, norepi_ug_kg_min]
    action_divergence: float         # ||a_ai - a_doc||
    estimated_reward_gain: float     # R_ai - R_doc


class CounterfactualTransferEstimator:
    """
    Computes off-policy counterfactual trajectories and evaluation scores on historical datasets.
    """

    def __init__(self, patient: PatientProfile):
        self.patient = patient
        
        # Internal parallel digital twins tracking doctor vs AI PK states
        self.sim_doctor = PatientSimulator(patient)
        self.sim_ai = PatientSimulator(patient)

    def evaluate_step(
        self,
        time_sec: float,
        dataset_bis: float,
        dataset_map: float,
        doctor_action: np.ndarray,
        ai_action: np.ndarray,
        dt_seconds: float = 1.0
    ) -> CounterfactualProjection:
        """
        Advances the counterfactual sensitivity estimator by dt_seconds.
        """
        doc_u = np.maximum(0.0, doctor_action)
        ai_u = np.maximum(0.0, ai_action)

        # 1. Step PK models under doctor vs AI delivery
        v_doc = self.sim_doctor.step(
            propofol_rate_mg_kg_h=float(doc_u[0]),
            remifentanil_rate_ug_kg_min=float(doc_u[1]),
            norepinephrine_rate_ug_kg_min=float(doc_u[2]),
            dt_seconds=dt_seconds
        )

        v_ai = self.sim_ai.step(
            propofol_rate_mg_kg_h=float(ai_u[0]),
            remifentanil_rate_ug_kg_min=float(ai_u[1]),
            norepinephrine_rate_ug_kg_min=float(ai_u[2]),
            dt_seconds=dt_seconds
        )

        # 2. Transfer differential: delta between AI and doctor digital twin
        delta_bis_transfer = v_ai.bis - v_doc.bis
        delta_map_transfer = v_ai.map_mmhg - v_doc.map_mmhg

        # 3. Apply transfer to historical ground-truth dataset vitals
        # Guard against unbounded noise propagation using soft saturation
        cf_bis = float(np.clip(dataset_bis + delta_bis_transfer, 0.0, 100.0))
        cf_map = float(np.clip(dataset_map + delta_map_transfer, 30.0, 160.0))

        # 4. Compute Off-Policy Reward Comparison
        # Positive score means AI projected vitals closer to therapeutic target (BIS=50, MAP=75)
        doc_bis_err = abs(dataset_bis - 50.0)
        ai_bis_err = abs(cf_bis - 50.0)
        
        doc_map_pen = max(0.0, 65.0 - dataset_map) * 2.0
        ai_map_pen = max(0.0, 65.0 - cf_map) * 2.0

        r_doc = -(doc_bis_err + doc_map_pen)
        r_ai = -(ai_bis_err + ai_map_pen)
        reward_gain = float(r_ai - r_doc)

        action_divergence = float(np.linalg.norm(ai_u - doc_u))

        return CounterfactualProjection(
            time_seconds=time_sec,
            dataset_bis=dataset_bis,
            counterfactual_bis=cf_bis,
            dataset_map=dataset_map,
            counterfactual_map=cf_map,
            doctor_action=doc_u,
            ai_action=ai_u,
            action_divergence=action_divergence,
            estimated_reward_gain=reward_gain
        )

    def reset(self) -> None:
        self.sim_doctor.reset()
        self.sim_ai.reset()
