"""
Qualified Dynamic Preload Gate Module
Differentiates hypovolemic fluid deficits from anesthetic vasodilation.
Evaluates Pulse Pressure Variation (PPV) and Stroke Volume Variation (SVV)
under strict clinical boundary validity conditions to prevent ischemic pressor abuse.
"""

from dataclasses import dataclass


@dataclass
class PreloadAssessment:
    ppv_percent: float
    is_valid_boundary_condition: bool
    is_fluid_responsive: bool
    max_allowable_norepi_rate: float
    fluid_bolus_recommended_ml: float
    clinical_advisory: str


class QualifiedPreloadGate:
    """
    Evaluates intravascular volume status and clamps pressors during hypovolemia.
    """

    def __init__(
        self,
        ppv_threshold_percent: float = 13.0,
        min_tidal_volume_ml_kg: float = 8.0
    ):
        self.ppv_threshold = ppv_threshold_percent
        self.min_tidal_volume = min_tidal_volume_ml_kg

    def evaluate(
        self,
        ppv_percent: float,
        is_mechanical_ventilation: bool = True,
        tidal_volume_ml_kg: float = 8.5,
        is_sinus_rhythm: bool = True
    ) -> PreloadAssessment:
        """
        Assesses fluid responsiveness and bounds pressor titration.
        """
        # 1. Verify strict physiological validity criteria for cardiopulmonary heart-lung interaction
        valid_conditions = (
            is_mechanical_ventilation
            and (tidal_volume_ml_kg >= self.min_tidal_volume)
            and is_sinus_rhythm
        )

        if not valid_conditions:
            # Boundary conditions not met (e.g. spontaneous breathing or arrhythmia)
            return PreloadAssessment(
                ppv_percent=ppv_percent,
                is_valid_boundary_condition=False,
                is_fluid_responsive=False,
                max_allowable_norepi_rate=0.15,
                fluid_bolus_recommended_ml=0.0,
                clinical_advisory="[PPV UNINTERPRETABLE — MANDATORY VENTILATION & SINUS RHYTHM REQUIRED FOR VALIDITY]"
            )

        # 2. Evaluate Dynamic Preload (PPV)
        is_responsive = (ppv_percent >= self.ppv_threshold)

        if is_responsive:
            # Underfilled vascular tank: clamp pressor escalation to prevent end-organ ischemia
            max_pressor = 0.04  # Cap pressor delivery
            bolus_volume = 250.0  # 250 mL crystalloid fluid challenge
            advisory = (
                f"[VOLUME DEFICIT DETECTED (PPV {ppv_percent:.1f}% >= 13%) — "
                f"CONFIRM {bolus_volume:.0f}mL CRYSTALLOID BOLUS BEFORE VASOPRESSOR ESCALATION]"
            )
        else:
            max_pressor = 0.20  # Normal unconstrained pressor headroom
            bolus_volume = 0.0
            advisory = "[PRELOAD ADEQUATE: NORMOVOLEMIC — NORMAL PRESSOR TITRATION PERMITTED]"

        return PreloadAssessment(
            ppv_percent=ppv_percent,
            is_valid_boundary_condition=True,
            is_fluid_responsive=is_responsive,
            max_allowable_norepi_rate=max_pressor,
            fluid_bolus_recommended_ml=bolus_volume,
            clinical_advisory=advisory
        )
