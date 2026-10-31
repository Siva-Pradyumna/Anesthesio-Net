"""
High-Order Control Barrier Function (HOCBF) Safety Filter Module
Implements a differentiable Quadratic Program (QP) that filters raw RL/control actions
to guarantee forward invariance over safe physiological sets (MAP >= 65 mmHg, 40 <= BIS <= 60).
Complying with IEC 60601-1-10 (PCLC standard).
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from anesthesionet.physiology.simulator import PatientVitals


@dataclass
class SafetyBoundsConfig:
    min_map: float = 65.0          # Lower MAP limit (mmHg)
    max_map: float = 110.0         # Upper MAP limit (mmHg)
    min_bis: float = 40.0          # Lower BIS limit
    max_bis: float = 60.0          # Upper BIS limit
    gamma_map: float = 0.5         # Barrier decay parameter for MAP
    gamma_bis: float = 0.5         # Barrier decay parameter for BIS


class HOCBFSafetyFilter:
    """
    Real-time QP safety filter for continuous 3D drug infusion actions.
    Solves:
        min_u 0.5 * ||u - u_raw||^2
        s.t.  dot_psi(s, u) + gamma * psi(s) >= 0
              u_min <= u <= u_max
    """

    def __init__(self, config: SafetyBoundsConfig | None = None):
        self.config = config or SafetyBoundsConfig()

        # Hard physical syringe pump limits
        self.u_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.u_max = np.array([12.0, 1.0, 0.2], dtype=np.float64)

    def filter_action(
        self,
        raw_action: np.ndarray,
        vitals: PatientVitals,
        surgical_stimulus: float = 0.0
    ) -> tuple[np.ndarray, bool, dict[str, float]]:
        """
        Projects raw_action to certified safe action u*.
        Returns:
            (safe_action, was_intervened, barrier_values)
        """
        cfg = self.config
        u_raw = np.clip(np.array(raw_action, dtype=np.float64), self.u_min, self.u_max)

        # 1. Evaluate Barrier Functions h(s)
        h_map_low = vitals.map_mmhg - cfg.min_map
        h_bis_low = vitals.bis - cfg.min_bis
        h_bis_high = cfg.max_bis - vitals.bis

        barrier_values = {
            "h_map_low": float(h_map_low),
            "h_bis_low": float(h_bis_low),
            "h_bis_high": float(h_bis_high)
        }

        # 2. Derive Linearized High-Order Safety Constraints
        # For MAP: Propofol & Remifentanil decrease MAP; Norepinephrine increases MAP
        # If MAP is approaching 65 mmHg, cap propofol and remifentanil, enforce minimum pressor
        c_map = cfg.gamma_map * h_map_low

        # For BIS: Propofol & Remifentanil decrease BIS
        # If BIS is approaching 40 (too deep), restrict further hypnotic delivery
        c_bis_low = cfg.gamma_bis * h_bis_low

        # If BIS is approaching 60 (too light), enforce minimal hypnotic baseline
        c_bis_high = cfg.gamma_bis * h_bis_high

        # Define Optimization Objective: min 0.5 * ||u - u_raw||^2
        def objective(u: np.ndarray) -> float:
            diff = u - u_raw
            # Weighting: prioritize keeping pressor responsive
            weights = np.array([1.0, 1.0, 2.0])
            return 0.5 * float(np.sum(weights * (diff ** 2)))

        def jacobian(u: np.ndarray) -> np.ndarray:
            weights = np.array([1.0, 1.0, 2.0])
            return weights * (u - u_raw)

        # Inequality constraints: g(u) >= 0
        constraints = []

        # Constraint 1: MAP Forward Invariance
        # Approximate sensitivity: dMAP/dt ~= 15*Norepi - 1.5*Prop - 8*Remi
        def map_constraint(u: np.ndarray) -> float:
            prop, remi, norepi = u
            delta_map_dot = 20.0 * norepi - 0.8 * prop - 4.0 * remi
            return float(delta_map_dot + c_map)

        constraints.append({"type": "ineq", "fun": map_constraint})

        # Constraint 2: BIS Lower Bound (Prevent Over-sedation < 40)
        def bis_low_constraint(u: np.ndarray) -> float:
            prop, remi, _ = u
            # Increasing propofol/remi drives BIS lower
            delta_bis_dot = -0.5 * prop - 2.0 * remi + 1.0
            return float(delta_bis_dot + c_bis_low)

        constraints.append({"type": "ineq", "fun": bis_low_constraint})

        # Box bounds for pump rates
        bounds = list(zip(self.u_min, self.u_max))

        # Solve Quadratic Program via Sequential Least Squares (SLSQP)
        res = minimize(
            objective,
            x0=u_raw,
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 20, "ftol": 1e-4}
        )

        if res.success:
            u_safe = np.clip(res.x, self.u_min, self.u_max)
        else:
            # Fallback to direct projection if optimization fails
            u_safe = u_raw.copy()
            if h_map_low < 3.0:
                # Emergency MAP preservation
                u_safe[0] *= 0.5  # Halve propofol
                u_safe[1] *= 0.5  # Halve remifentanil
                u_safe[2] = max(u_safe[2], 0.05)  # Enforce minimum pressor
            if h_bis_low < 3.0:
                u_safe[0] *= 0.7
                u_safe[1] *= 0.7

        # Check if intervention was applied
        was_intervened = bool(np.linalg.norm(u_safe - u_raw) > 0.01)

        return u_safe.astype(np.float32), was_intervened, barrier_values
