"""
Minto/Bouillon Pharmacodynamic Synergy Response Surface Module
Calculates Bispectral Index (BIS) depth of hypnosis from the non-linear interaction
of Propofol (Ce_prop, ug/mL) and Remifentanil (Ce_remi, ng/mL).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SynergyParameters:
    """
    Pharmacodynamic response surface parameters for Propofol + Remifentanil.
    """
    EC50_prop: float = 3.5    # Propofol EC50 alone (ug/mL)
    EC50_remi: float = 8.0    # Remifentanil EC50 alone (ng/mL)
    beta: float = 1.3         # Supra-additive synergy interaction parameter (Greco/Bouillon model)
    gamma: float = 2.8        # Hill steepness coefficient
    E0_bis: float = 98.0      # Baseline awake BIS value (0 - 100)


class PharmacodynamicSynergy:
    """
    Computes depth of hypnosis (BIS) and sedation levels given effect-site concentrations.
    """

    def __init__(self, params: SynergyParameters | None = None):
        self.params = params or SynergyParameters()

    def compute_bis(self, Ce_prop_ug_ml: float, Ce_remi_ng_ml: float) -> float:
        """
        Calculates predicted Bispectral Index (BIS) from 0 to 100.
        Ce_prop_ug_ml: Propofol effect-site concentration (ug/mL)
        Ce_remi_ng_ml: Remifentanil effect-site concentration (ng/mL)
        """
        p = self.params

        # Guard against negative effect concentrations
        ce_p = max(0.0, float(Ce_prop_ug_ml))
        ce_r = max(0.0, float(Ce_remi_ng_ml))

        if ce_p < 1e-4 and ce_r < 1e-4:
            return p.E0_bis

        # Normalized drug potencies
        u_a = ce_p / p.EC50_prop
        u_b = ce_r / p.EC50_remi

        sum_u = u_a + u_b
        if sum_u <= 0.0:
            return p.E0_bis

        # Ratio of Propofol to total drug potency
        theta = u_a / sum_u

        # Bouillon / Minto interaction factor
        # When beta > 0, U50 < 1 which represents supra-additive synergy
        u50_theta = 1.0 - p.beta * theta * (1.0 - theta)
        u50_theta = max(0.1, u50_theta)  # Numerical stability floor

        # Normalized synergistic drug effect
        eff_ratio = sum_u / u50_theta

        # Sigmoidal Hill transformation
        eff_powered = eff_ratio ** p.gamma
        fractional_effect = eff_powered / (1.0 + eff_powered)

        # Resulting BIS
        bis = p.E0_bis * (1.0 - fractional_effect)
        return float(np.clip(bis, 0.0, 100.0))

    def evaluate_optimal_remi_prior(self, Ce_prop_ug_ml: float) -> float:
        """
        Returns the clinically optimal Remifentanil effect-site concentration (ng/mL)
        for a given Propofol concentration to maintain a balanced hypnotic-opioid ratio.
        Used as an action regularizer in the SAC RL policy.
        """
        # Clinical target: under general anesthesia (BIS 45-55),
        # optimal Ce_remi typically sits between 2.0 and 4.0 ng/mL
        if Ce_prop_ug_ml <= 0.5:
            return 0.0
        # Optimal synergistic ridge
        return float(np.clip(3.5 - 0.4 * Ce_prop_ug_ml, 1.5, 5.0))
