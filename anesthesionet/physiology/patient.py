"""
Patient Covariate and Demographics Module
Calculates derived clinical parameters (BMI, Lean Body Mass / Fat-Free Mass)
using clinically validated allometric scaling formulas.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PatientProfile:
    """
    Patient demographics and baseline physiological parameters.
    """
    age_years: float               # Age in years
    weight_kg: float                # Total body weight in kg
    height_cm: float                # Height in cm
    sex: Literal["male", "female"]  # Biological sex

    # Baseline physiological states (awake, unanesthetized)
    baseline_map: float = 90.0      # Baseline Mean Arterial Pressure (mmHg)
    baseline_hr: float = 75.0       # Baseline Heart Rate (bpm)
    baseline_bis: float = 98.0      # Baseline Bispectral Index (0-100)

    @property
    def bmi(self) -> float:
        """Body Mass Index (kg/m^2)"""
        height_m = self.height_cm / 100.0
        return self.weight_kg / (height_m ** 2)

    @property
    def fat_free_mass_janmahasatian(self) -> float:
        """
        Fat-Free Mass (FFM) in kg according to the Janmahasatian (2005) formula,
        recommended by Eleveld for allometric scaling in both normal and obese populations.
        """
        bmi = self.bmi
        if self.sex == "male":
            return (9270 * self.weight_kg) / (6680 + 216 * bmi)
        else:
            return (9270 * self.weight_kg) / (8780 + 244 * bmi)

    @property
    def lean_body_mass_james(self) -> float:
        """
        Legacy James formula for Lean Body Mass (LBM) in kg (used in Schnider model).
        Note: James formula exhibits paradoxical negative values in extreme obesity (BMI > 42).
        """
        bmi = self.bmi
        if self.sex == "male":
            lbm = 1.1 * self.weight_kg - 128 * ((self.weight_kg / self.height_cm) ** 2)
        else:
            lbm = 1.07 * self.weight_kg - 148 * ((self.weight_kg / self.height_cm) ** 2)
        
        # Guard against James formula breakdown in severe obesity
        return max(lbm, 0.5 * self.weight_kg)
