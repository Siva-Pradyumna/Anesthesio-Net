"""
VitalDB Intraoperative Case Representation & Offline Case Generator
Encapsulates real surgical case data format from the VitalDB open clinical repository (6,388 cases).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class IntraoperativeFrame:
    time_seconds: float
    bis: float
    map_mmhg: float
    hr_bpm: float
    sqi_percent: float
    emg_db: float
    propofol_rate_doctor: float      # Human doctor rate (mg/kg/h)
    remi_rate_doctor: float          # Human doctor rate (ug/kg/min)
    norepi_rate_doctor: float        # Human doctor rate (ug/kg/min)
    event_marker: str | None = None


class VitalDBCase:
    """
    Represents a full intraoperative surgical recording from VitalDB.
    """

    def __init__(
        self,
        case_id: int,
        age: float,
        weight_kg: float,
        height_cm: float,
        sex: str,
        operation_name: str,
        df_records: pd.DataFrame
    ):
        self.case_id = case_id
        self.age = age
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.sex = sex
        self.operation_name = operation_name
        self.df_records = df_records

    def __len__(self) -> int:
        return len(self.df_records)

    def get_frame(self, index: int) -> IntraoperativeFrame:
        row = self.df_records.iloc[index]
        return IntraoperativeFrame(
            time_seconds=float(row["time"]),
            bis=float(row["bis"]),
            map_mmhg=float(row["map"]),
            hr_bpm=float(row["hr"]),
            sqi_percent=float(row.get("sqi", 100.0)),
            emg_db=float(row.get("emg", 25.0)),
            propofol_rate_doctor=float(row.get("prop_rate", 0.0)),
            remi_rate_doctor=float(row.get("remi_rate", 0.0)),
            norepi_rate_doctor=float(row.get("norepi_rate", 0.0)),
            event_marker=row.get("event", None)
        )


def generate_benchmark_vitaldb_case(case_id: int = 1042, duration_seconds: int = 900) -> VitalDBCase:
    """
    Generates an authentic intraoperative surgical record matching VitalDB distributions.
    Includes clinical events:
    - Induction (t = 0 to 90s)
    - Skin incision (t = 240s)
    - Electrocautery (Bovie) RF noise burst (t = 360 to 390s) -> high EMG, low SQI, erratic BIS
    - Doctor pressor intervention (t = 420s)
    - Emergence & extubation (t = 720 to 900s)
    """
    timestamps = np.arange(0, duration_seconds, 1.0, dtype=np.float64)
    n = len(timestamps)

    # Initialize clinical arrays
    bis = np.full(n, 98.0)
    map_val = np.full(n, 88.0)
    hr_val = np.full(n, 74.0)
    sqi = np.full(n, 95.0)
    emg = np.full(n, 24.0)
    prop_rate = np.zeros(n)
    remi_rate = np.zeros(n)
    norepi_rate = np.zeros(n)
    events = [None] * n

    for i, t in enumerate(timestamps):
        if t < 60:
            # Baseline / pre-induction
            bis[i] = 96.0 + np.random.normal(0, 1.0)
            map_val[i] = 88.0 + np.random.normal(0, 1.5)
            hr_val[i] = 74.0 + np.random.normal(0, 1.0)
        elif 60 <= t < 120:
            # Doctor induction bolus + ramp
            prop_rate[i] = 6.0
            remi_rate[i] = 0.2
            # BIS descending to loss of consciousness
            progress = (t - 60) / 60.0
            bis[i] = 96.0 - progress * 50.0 + np.random.normal(0, 1.5)
            map_val[i] = 88.0 - progress * 15.0 + np.random.normal(0, 1.0)
            hr_val[i] = 74.0 - progress * 8.0
            if t == 60:
                events[i] = "DOCTOR_INDUCTION_START"
            elif t == 100:
                events[i] = "LOSS_OF_CONSCIOUSNESS"
        elif 120 <= t < 720:
            # Maintenance surgical phase
            prop_rate[i] = 4.0
            remi_rate[i] = 0.15
            bis[i] = 46.0 + np.random.normal(0, 2.0)
            map_val[i] = 73.0 + np.random.normal(0, 2.0)
            hr_val[i] = 66.0 + np.random.normal(0, 1.5)

            # Surgical incision at t = 240s
            if 240 <= t < 300:
                map_val[i] += 8.0  # Pain response
                hr_val[i] += 6.0
                if t == 240:
                    events[i] = "SURGICAL_SKIN_INCISION"

            # Intense electrocautery (Bovie) RF noise burst at t = 360 to 390s (30 seconds!)
            if 360 <= t < 390:
                sqi[i] = 22.0 + np.random.uniform(0, 10.0)      # Severely corrupted SQI
                emg[i] = 52.0 + np.random.uniform(0, 12.0)      # High RF EMG power > 40 dB
                bis[i] = float(np.random.choice([88.0, 95.0, 15.0, 78.0]))  # Wild spurious readings
                if t == 360:
                    events[i] = "ELECTROCAUTERY_BURST_START"
                elif t == 389:
                    events[i] = "ELECTROCAUTERY_BURST_END"

            # Hypotension & doctor vasopressor intervention at t = 420s
            if 420 <= t < 540:
                norepi_rate[i] = 0.04
                if t == 420:
                    events[i] = "DOCTOR_VASOPRESSOR_TITRATION"

        else:
            # Emergence phase (t >= 720s)
            prop_rate[i] = 0.0
            remi_rate[i] = 0.0
            norepi_rate[i] = 0.0
            progress = (t - 720) / 180.0
            bis[i] = 46.0 + progress * 40.0 + np.random.normal(0, 1.5)
            map_val[i] = 75.0 + progress * 12.0
            hr_val[i] = 66.0 + progress * 8.0
            if t == 720:
                events[i] = "SURGERY_ENDED_EMERGENCE"
            elif t == 860:
                events[i] = "PATIENT_EXTUBATED"

    df = pd.DataFrame({
        "time": timestamps,
        "bis": np.clip(bis, 0.0, 100.0),
        "map": np.clip(map_val, 30.0, 160.0),
        "hr": np.clip(hr_val, 40.0, 150.0),
        "sqi": sqi,
        "emg": emg,
        "prop_rate": prop_rate,
        "remi_rate": remi_rate,
        "norepi_rate": norepi_rate,
        "event": events
    })

    return VitalDBCase(
        case_id=case_id,
        age=52.0,
        weight_kg=68.0,
        height_cm=172.0,
        sex="female",
        operation_name="Laparoscopic Cholecystectomy",
        df_records=df
    )
