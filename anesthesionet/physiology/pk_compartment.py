"""
Three-Compartment Pharmacokinetic State-Space Module
Implements coupled linear differential equations with effect-site equilibration
for anesthetic drugs (Propofol, Remifentanil).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PKParameters:
    """
    Pharmacokinetic parameters for a 3-compartment mammillary model.
    Volumes in Liters (L), Clearances in L/min, ke0 in 1/min.
    """
    V1: float    # Central compartment volume (L)
    V2: float    # Rapid peripheral compartment volume (L)
    V3: float    # Slow peripheral compartment volume (L)
    Cl1: float   # Metabolic clearance from V1 (L/min)
    Cl2: float   # Inter-compartmental clearance between V1 and V2 (L/min)
    Cl3: float   # Inter-compartmental clearance between V1 and V3 (L/min)
    ke0: float   # Effect-site elimination rate constant (1/min)

    @property
    def k10(self) -> float:
        """Elimination rate constant from central compartment (1/min)"""
        return self.Cl1 / self.V1

    @property
    def k12(self) -> float:
        """Transfer rate constant from central to rapid peripheral (1/min)"""
        return self.Cl2 / self.V1

    @property
    def k21(self) -> float:
        """Transfer rate constant from rapid peripheral to central (1/min)"""
        return self.Cl2 / self.V2

    @property
    def k13(self) -> float:
        """Transfer rate constant from central to slow peripheral (1/min)"""
        return self.Cl3 / self.V1

    @property
    def k31(self) -> float:
        """Transfer rate constant from slow peripheral to central (1/min)"""
        return self.Cl3 / self.V3


class ThreeCompartmentPK:
    """
    Simulates drug mass and concentration dynamics across 3 pharmacokinetic compartments
    plus an effect-site biophase compartment.
    """

    def __init__(self, params: PKParameters):
        self.params = params
        
        # State vector: [x1 (mass in V1), x2 (mass in V2), x3 (mass in V3), Ce (effect-site conc)]
        self.state = np.zeros(4, dtype=np.float64)

    @property
    def mass_v1(self) -> float:
        return float(self.state[0])

    @property
    def mass_v2(self) -> float:
        return float(self.state[1])

    @property
    def mass_v3(self) -> float:
        return float(self.state[2])

    @property
    def Cp(self) -> float:
        """Central plasma concentration = mass_V1 / V1"""
        return float(self.state[0] / self.params.V1)

    @property
    def Ce(self) -> float:
        """Effect-site concentration"""
        return float(self.state[3])

    def _derivatives(self, state: np.ndarray, infusion_rate_per_min: float) -> np.ndarray:
        """
        Computes dx/dt where t is in minutes.
        state = [x1, x2, x3, Ce]
        infusion_rate_per_min: Drug delivery rate in mass units per minute.
        """
        x1, x2, x3, Ce = state
        p = self.params

        # dx1/dt = -(k10 + k12 + k13)*x1 + k21*x2 + k31*x3 + I(t)
        dx1_dt = -(p.k10 + p.k12 + p.k13) * x1 + p.k21 * x2 + p.k31 * x3 + infusion_rate_per_min
        
        # dx2/dt = k12*x1 - k21*x2
        dx2_dt = p.k12 * x1 - p.k21 * x2
        
        # dx3/dt = k13*x1 - k31*x3
        dx3_dt = p.k13 * x1 - p.k31 * x3
        
        # dCe/dt = ke0 * (Cp - Ce) = ke0 * (x1 / V1 - Ce)
        dCe_dt = p.ke0 * ((x1 / p.V1) - Ce)

        return np.array([dx1_dt, dx2_dt, dx3_dt, dCe_dt], dtype=np.float64)

    def step(self, infusion_rate_per_sec: float, dt_seconds: float = 0.1) -> tuple[float, float]:
        """
        Advances the PK state by dt_seconds using 4th-order Runge-Kutta integration.
        infusion_rate_per_sec: mass units per second (e.g. mg/s or ug/s).
        Returns:
            Tuple of (Cp, Ce) in mass units / L (or ug/mL).
        """
        dt_min = dt_seconds / 60.0
        infusion_rate_per_min = infusion_rate_per_sec * 60.0

        # RK4 Integration
        k1 = self._derivatives(self.state, infusion_rate_per_min)
        k2 = self._derivatives(self.state + 0.5 * dt_min * k1, infusion_rate_per_min)
        k3 = self._derivatives(self.state + 0.5 * dt_min * k2, infusion_rate_per_min)
        k4 = self._derivatives(self.state + dt_min * k3, infusion_rate_per_min)

        self.state += (dt_min / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Guard against minor numerical floating-point negative values
        self.state = np.maximum(self.state, 0.0)

        return self.Cp, self.Ce

    def bolus(self, dose_amount: float) -> None:
        """Administers an instantaneous intravenous bolus directly into central compartment V1."""
        self.state[0] += max(0.0, dose_amount)

    def reset(self) -> None:
        """Clears all compartment drug amounts to zero."""
        self.state.fill(0.0)
