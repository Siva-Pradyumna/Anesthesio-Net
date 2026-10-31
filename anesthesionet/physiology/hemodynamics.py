"""
Coupled Hemodynamics Module
Models Mean Arterial Pressure (MAP, mmHg) and Heart Rate (HR, bpm) responses
to Propofol vasodilation, Remifentanil vagal tone, Norepinephrine vasoconstriction,
dynamic preload status (PPV), and surgical noxious incision stimuli.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class HemodynamicParameters:
    baseline_map: float = 90.0        # Baseline MAP (mmHg)
    baseline_hr: float = 75.0         # Baseline HR (bpm)
    baseline_ppv: float = 9.0         # Baseline Pulse Pressure Variation (%)
    
    # Drug sensitivity coefficients
    prop_map_max_drop: float = 32.0   # Max MAP reduction from Propofol (mmHg)
    prop_ec50_map: float = 3.0        # Ce_prop (ug/mL) at half-maximal MAP drop
    
    remi_hr_max_drop: float = 25.0    # Max HR reduction from Remifentanil (bpm)
    remi_ec50_hr: float = 3.0         # Ce_remi (ng/mL) at half-maximal HR drop
    
    norepi_map_max_gain: float = 50.0 # Max MAP elevation from Norepinephrine (mmHg)
    norepi_ec50_map: float = 0.08     # Ce_norepi (ug/kg/min) at half-maximal gain
    norepi_ke0: float = 1.0           # Norepinephrine effect-site equilibration constant (1/min)


class CoupledHemodynamics:
    """
    Simulates real-time hemodynamic changes governed by drug effect concentrations
    and surgical stimulation.
    """

    def __init__(self, params: HemodynamicParameters | None = None):
        self.params = params or HemodynamicParameters()
        
        # State variables
        self.current_map = self.params.baseline_map
        self.current_hr = self.params.baseline_hr
        self.current_ppv = self.params.baseline_ppv
        
        # Norepinephrine effect-site concentration tracker
        self.ce_norepi = 0.0

    def step(
        self,
        Ce_prop_ug_ml: float,
        Ce_remi_ng_ml: float,
        norepi_infusion_ug_kg_min: float,
        surgical_noxious_stimulus: float = 0.0,
        fluid_deficit_factor: float = 1.0,
        dt_seconds: float = 0.1
    ) -> tuple[float, float, float]:
        """
        Advances hemodynamic state by dt_seconds.
        Args:
            Ce_prop_ug_ml: Propofol effect-site conc (ug/mL)
            Ce_remi_ng_ml: Remifentanil effect-site conc (ng/mL)
            norepi_infusion_ug_kg_min: Norepinephrine delivery rate (ug/kg/min)
            surgical_noxious_stimulus: Surgical pain intensity (0.0 to 1.0)
            fluid_deficit_factor: 1.0 = normovolemic, > 1.2 = hypovolemic
            dt_seconds: integration step size in seconds

        Returns:
            Tuple of (MAP in mmHg, HR in bpm, PPV in %)
        """
        p = self.params
        dt_min = dt_seconds / 60.0

        # 1. Update Norepinephrine effect-site concentration (1st-order biophase lag)
        d_ce_norepi = p.norepi_ke0 * (norepi_infusion_ug_kg_min - self.ce_norepi)
        self.ce_norepi += d_ce_norepi * dt_min
        self.ce_norepi = max(0.0, self.ce_norepi)

        # 2. Propofol Vasodilation & Myocardial Depression on MAP
        prop_effect = (Ce_prop_ug_ml ** 1.8) / ((p.prop_ec50_map ** 1.8) + (Ce_prop_ug_ml ** 1.8) + 1e-6)
        delta_map_prop = -p.prop_map_max_drop * prop_effect

        # 3. Norepinephrine alpha-1 Vasoconstriction on MAP
        norepi_effect = (self.ce_norepi ** 1.5) / ((p.norepi_ec50_map ** 1.5) + (self.ce_norepi ** 1.5) + 1e-6)
        delta_map_norepi = p.norepi_map_max_gain * norepi_effect

        # 4. Surgical Noxious Stimulation (Incision Pain)
        # Remifentanil blunts the hemodynamic response to surgical stimulation
        opioid_blockade = Ce_remi_ng_ml / (Ce_remi_ng_ml + 2.0 + 1e-6)
        unblocked_stimulus = surgical_noxious_stimulus * (1.0 - opioid_blockade)
        delta_map_pain = unblocked_stimulus * 25.0
        delta_hr_pain = unblocked_stimulus * 30.0

        # 5. Remifentanil Central Vagal Bradycardia on HR
        remi_hr_effect = (Ce_remi_ng_ml ** 1.5) / ((p.remi_ec50_hr ** 1.5) + (Ce_remi_ng_ml ** 1.5) + 1e-6)
        delta_hr_remi = -p.remi_hr_max_drop * remi_hr_effect

        # 6. Baroreceptor Reflex (Hypotension triggers tachycardia)
        target_map = (p.baseline_map + delta_map_prop + delta_map_norepi + delta_map_pain) / fluid_deficit_factor
        map_error = p.baseline_map - target_map
        baroreflex_hr = 0.25 * map_error if map_error > 0 else 0.0

        # Calculate final targets
        target_hr = p.baseline_hr + delta_hr_remi + delta_hr_pain + baroreflex_hr

        # Smooth dynamic transitions (first-order inertia)
        tau_map = 0.05  # Smoothing factor
        self.current_map += (target_map - self.current_map) * tau_map
        self.current_hr += (target_hr - self.current_hr) * tau_map

        # Dynamic Pulse Pressure Variation (PPV)
        # PPV climbs when fluid deficit increases or stroke volume swings widen
        self.current_ppv = float(np.clip(p.baseline_ppv * fluid_deficit_factor + (map_error * 0.1), 4.0, 30.0))

        # Physiological bounds
        self.current_map = float(np.clip(self.current_map, 20.0, 180.0))
        self.current_hr = float(np.clip(self.current_hr, 30.0, 180.0))

        return self.current_map, self.current_hr, self.current_ppv

    def reset(self) -> None:
        """Resets hemodynamics to baseline awake status."""
        self.current_map = self.params.baseline_map
        self.current_hr = self.params.baseline_hr
        self.current_ppv = self.params.baseline_ppv
        self.ce_norepi = 0.0
