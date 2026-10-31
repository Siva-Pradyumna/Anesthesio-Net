"""
Clinical and Engineering Evaluation Metrics Module
Implements standard clinical anesthesia performance benchmarks:
- Varvel Metrics: MDPE (Bias), MDAPE (Inaccuracy), Wobble (Divergence)
- Time in Therapeutic Range (TITR) for BIS (40-60) and MAP (65-100 mmHg)
- Safety Filter Interception Efficiency & Circulatory Preservation Score
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ClinicalPerformanceReport:
    # Varvel Metrics on BIS
    mdpe_bis_percent: float          # Median Performance Error (Bias) -> Ideal: -10% to +10%
    mdape_bis_percent: float         # Median Absolute Performance Error -> Ideal: < 15%
    wobble_bis_percent: float        # Intra-individual variability -> Ideal: < 15%
    
    # Therapeutic Window Compliance
    titr_bis_percent: float          # % time BIS in target [40, 60] -> Ideal: > 80%
    titr_map_percent: float          # % time MAP in target [65, 100] -> Ideal: > 85%
    
    # Induction & Emergence Timing
    time_to_loc_seconds: float       # Time to Loss of Consciousness (BIS < 60) -> Ideal: < 120s
    hysteresis_dwell_seconds: float  # Continuous dwell before incision -> Target: >= 30s
    time_to_emergence_seconds: float # Time from surgery end to BIS >= 80 -> Ideal: 300-600s
    
    # Safety Interlocks
    lethal_actions_injected: int     # Total stress-test dangerous commands
    lethal_actions_intercepted: int  # Overrides successfully neutralized by HOCBF
    circulatory_collapse_events: int # Critical events (MAP < 50 or HR < 40) -> Target: 0


def compute_varvel_metrics(measured: np.ndarray, target: float = 50.0) -> tuple[float, float, float]:
    """
    Computes Varvel clinical metrics for closed-loop anesthesia controllers:
    - PE_i = ((measured_i - target) / target) * 100%
    - MDPE = median(PE)
    - MDAPE = median(|PE|)
    - Wobble = median(|PE - MDPE|)
    """
    if len(measured) == 0:
        return 0.0, 0.0, 0.0
    
    pe = ((measured - target) / target) * 100.0
    mdpe = float(np.median(pe))
    mdape = float(np.median(np.abs(pe)))
    wobble = float(np.median(np.abs(pe - mdpe)))
    
    return mdpe, mdape, wobble


def compute_time_in_range(values: np.ndarray, low: float, high: float) -> float:
    """Computes the percentage of time points falling within [low, high]."""
    if len(values) == 0:
        return 0.0
    in_range = np.logical_and(values >= low, values <= high)
    return float(np.mean(in_range) * 100.0)
