"""
Unified Mode A Patient Simulator (Digital Twin)
Couples multi-drug 3-compartment pharmacokinetics with synergistic pharmacodynamics
and hemodynamics into an interactive real-time simulation engine.
"""

from dataclasses import dataclass
from .patient import PatientProfile
from .pk_compartment import ThreeCompartmentPK
from .eleveld import compute_eleveld_propofol, compute_eleveld_remifentanil
from .synergy import PharmacodynamicSynergy
from .hemodynamics import CoupledHemodynamics


@dataclass
class PatientVitals:
    """Snapshot of real-time patient physiological and drug status."""
    time_seconds: float
    bis: float                   # Bispectral Index (0 - 100)
    map_mmhg: float              # Mean Arterial Pressure (mmHg)
    hr_bpm: float                # Heart Rate (bpm)
    ppv_percent: float           # Pulse Pressure Variation (%)
    
    # Effect-site and plasma concentrations
    cp_propofol_ug_ml: float     # Propofol Plasma Conc (ug/mL)
    ce_propofol_ug_ml: float     # Propofol Effect-site Conc (ug/mL)
    cp_remi_ng_ml: float         # Remifentanil Plasma Conc (ng/mL)
    ce_remi_ng_ml: float         # Remifentanil Effect-site Conc (ng/mL)
    
    # Syringe pump delivery rates
    propofol_rate_mg_kg_h: float
    remifentanil_rate_ug_kg_min: float
    norepinephrine_rate_ug_kg_min: float


class PatientSimulator:
    """
    Mode A Digital Twin Simulator solving patient physiology at 100 ms (10 Hz) steps.
    """

    def __init__(self, patient: PatientProfile):
        self.patient = patient
        
        # 1. Initialize Pharmacokinetics (Eleveld Models)
        self.prop_params = compute_eleveld_propofol(patient)
        self.remi_params = compute_eleveld_remifentanil(patient)
        
        self.pk_prop = ThreeCompartmentPK(self.prop_params)
        self.pk_remi = ThreeCompartmentPK(self.remi_params)
        
        # 2. Initialize Pharmacodynamic Synergy (Minto/Bouillon)
        self.pd_synergy = PharmacodynamicSynergy()
        
        # 3. Initialize Coupled Hemodynamics
        self.hemodynamics = CoupledHemodynamics()
        
        # Internal clock
        self.simulation_time_sec = 0.0

    def step(
        self,
        propofol_rate_mg_kg_h: float = 0.0,
        remifentanil_rate_ug_kg_min: float = 0.0,
        norepinephrine_rate_ug_kg_min: float = 0.0,
        surgical_stimulus: float = 0.0,
        fluid_deficit_factor: float = 1.0,
        dt_seconds: float = 0.1
    ) -> PatientVitals:
        """
        Advances the entire patient digital twin state by dt_seconds.
        """
        weight = self.patient.weight_kg

        # Convert clinical pump rates to mass units per second:
        # Propofol: (mg/kg/h * weight) / 3600 -> mg/s (mass in mg, V1 in L -> Cp in mg/L = ug/mL)
        prop_mg_per_sec = (max(0.0, propofol_rate_mg_kg_h) * weight) / 3600.0
        
        # Remifentanil: (ug/kg/min * weight) / 60 -> ug/s (mass in ug, V1 in L -> Cp in ug/L = ng/mL)
        remi_ug_per_sec = (max(0.0, remifentanil_rate_ug_kg_min) * weight) / 60.0

        # 1. Step Pharmacokinetics
        cp_prop, ce_prop = self.pk_prop.step(prop_mg_per_sec, dt_seconds=dt_seconds)
        cp_remi, ce_remi = self.pk_remi.step(remi_ug_per_sec, dt_seconds=dt_seconds)

        # 2. Step Pharmacodynamic Synergy (BIS)
        bis = self.pd_synergy.compute_bis(ce_prop, ce_remi)

        # 3. Step Coupled Hemodynamics (MAP, HR, PPV)
        map_val, hr_val, ppv_val = self.hemodynamics.step(
            Ce_prop_ug_ml=ce_prop,
            Ce_remi_ng_ml=ce_remi,
            norepi_infusion_ug_kg_min=max(0.0, norepinephrine_rate_ug_kg_min),
            surgical_noxious_stimulus=surgical_stimulus,
            fluid_deficit_factor=fluid_deficit_factor,
            dt_seconds=dt_seconds
        )

        self.simulation_time_sec += dt_seconds

        return PatientVitals(
            time_seconds=self.simulation_time_sec,
            bis=bis,
            map_mmhg=map_val,
            hr_bpm=hr_val,
            ppv_percent=ppv_val,
            cp_propofol_ug_ml=cp_prop,
            ce_propofol_ug_ml=ce_prop,
            cp_remi_ng_ml=cp_remi,
            ce_remi_ng_ml=ce_remi,
            propofol_rate_mg_kg_h=propofol_rate_mg_kg_h,
            remifentanil_rate_ug_kg_min=remifentanil_rate_ug_kg_min,
            norepinephrine_rate_ug_kg_min=norepinephrine_rate_ug_kg_min
        )

    def administer_propofol_bolus(self, dose_mg: float) -> None:
        """Administers an instantaneous intravenous bolus of Propofol (mg)."""
        self.pk_prop.bolus(dose_mg)

    def reset(self) -> None:
        """Resets the simulator to baseline unanesthetized state."""
        self.pk_prop.reset()
        self.pk_remi.reset()
        self.hemodynamics.reset()
        self.simulation_time_sec = 0.0
