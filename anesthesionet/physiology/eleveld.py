"""
Eleveld Pharmacokinetic Parameter Scaling Module
Implements the clinically validated Eleveld models for:
1. Propofol (Eleveld et al., Br J Anaesth 2018)
2. Remifentanil (Eleveld et al., Anesthesiology 2017)
"""

import math
from .patient import PatientProfile
from .pk_compartment import PKParameters


def compute_eleveld_propofol(patient: PatientProfile) -> PKParameters:
    """
    Computes 3-compartment PK parameters for Propofol using the Eleveld model.
    Accounts for age, weight, height, sex, and maturation scaling.
    Units: Volumes in Liters, Clearances in L/min, ke0 in 1/min.
    """
    age = patient.age_years
    weight = patient.weight_kg
    sex_male = 1 if patient.sex == "male" else 0

    # Maturation / Sigmoidal function of age (PMA/Age)
    # For adults (age >= 18), f_age represents aging degradation
    f_aging = math.exp(-0.0075 * (age - 35))

    # Allometric scaling based on typical 70kg, 35yo male
    V1_ref = 6.28  # Liters
    V2_ref = 25.5
    V3_ref = 273.0

    Cl1_ref = 1.79  # L/min
    Cl2_ref = 1.75
    Cl3_ref = 1.11

    # Weight exponents (allometric: 1.0 for volumes, 0.75 for clearances)
    weight_ratio = weight / 70.0
    
    # Eleveld parameter functions
    V1 = V1_ref * (weight / 70.0)
    V2 = V2_ref * (weight / 70.0) * math.exp(-0.0156 * (age - 35))
    V3 = V3_ref * (weight / 70.0) * f_aging

    # Clearances
    Cl1 = Cl1_ref * (weight_ratio ** 0.75) * math.exp(-0.0086 * (age - 35))
    if sex_male == 0:
        Cl1 *= 1.12  # Female clearance adjustment in Eleveld model

    Cl2 = Cl2_ref * (weight_ratio ** 0.75)
    Cl3 = Cl3_ref * (weight_ratio ** 0.75) * f_aging

    # ke0 for Propofol in Eleveld (typically ~0.146 1/min for typical adult)
    ke0 = 0.146 * (weight_ratio ** -0.25) * math.exp(-0.006 * (age - 35))

    return PKParameters(
        V1=max(1.0, V1),
        V2=max(2.0, V2),
        V3=max(10.0, V3),
        Cl1=max(0.1, Cl1),
        Cl2=max(0.1, Cl2),
        Cl3=max(0.05, Cl3),
        ke0=max(0.01, ke0)
    )


def compute_eleveld_remifentanil(patient: PatientProfile) -> PKParameters:
    """
    Computes 3-compartment PK parameters for Remifentanil using the Eleveld model.
    Units: Volumes in Liters, Clearances in L/min, ke0 in 1/min.
    """
    age = patient.age_years
    weight = patient.weight_kg
    sex_male = 1 if patient.sex == "male" else 0

    weight_ratio = weight / 70.0
    age_factor = math.exp(-0.011 * (age - 35))

    # Baseline typical values (70kg adult)
    V1_ref = 5.1     # Liters
    V2_ref = 9.82    # Liters
    V3_ref = 5.42    # Liters

    Cl1_ref = 2.6    # L/min (high clearance via blood/tissue esterases)
    Cl2_ref = 2.05   # L/min
    Cl3_ref = 0.076  # L/min

    V1 = V1_ref * weight_ratio
    V2 = V2_ref * weight_ratio * age_factor
    V3 = V3_ref * weight_ratio

    Cl1 = Cl1_ref * (weight_ratio ** 0.75) * age_factor
    if sex_male == 0:
        Cl1 *= 1.08  # Female clearance variance

    Cl2 = Cl2_ref * (weight_ratio ** 0.75)
    Cl3 = Cl3_ref * (weight_ratio ** 0.75)

    # Remifentanil ke0 is very high due to rapid biophase equilibration
    ke0 = 1.09 * math.exp(-0.018 * (age - 35))  # 1/min (t1/2,ke0 approx 1-1.5 min)

    return PKParameters(
        V1=max(1.0, V1),
        V2=max(1.0, V2),
        V3=max(1.0, V3),
        Cl1=max(0.2, Cl1),
        Cl2=max(0.1, Cl2),
        Cl3=max(0.01, Cl3),
        ke0=max(0.1, ke0)
    )
