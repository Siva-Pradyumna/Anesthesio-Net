"""
Signal Integrity & Electrosurgical Artifact Rejection Layer
Implements:
1. Dynamic velocity gate: |ΔBIS| / Δt > 20 units/s flags non-physiological spikes.
2. 5-second rolling median filter for high-frequency burst rejection.
3. Spectral Signal Quality Index (SQI < 50%) and EMG (> 40 dB) detection.
4. Automated "Sensor Hold Mode" failsafe under sustained electrocautery (Bovie) noise.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class FilteredTelemetry:
    time_seconds: float
    raw_bis: float
    filtered_bis: float
    raw_map: float
    filtered_map: float
    raw_hr: float
    filtered_hr: float
    sqi_percent: float
    emg_db: float
    is_artifact: bool
    sensor_hold_active: bool
    status_message: str


@dataclass
class FilterConfig:
    velocity_threshold_bis_per_sec: float = 20.0  # Max physiological BIS change rate
    min_sqi_threshold: float = 50.0               # Minimum valid SQI percentage
    max_emg_threshold_db: float = 40.0            # Max allowable EMG noise power (dB)
    median_window_seconds: float = 5.0            # Rolling median window
    hold_trigger_duration_seconds: float = 3.0    # Sustained noise duration to trigger hold


class SignalArtifactFilter:
    """
    Real-time noise and artifact suppression filter for intraoperative telemetry.
    """

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()
        
        # Rolling historical buffers for median filtering
        self.bis_buffer = deque()
        self.map_buffer = deque()
        self.hr_buffer = deque()

        # State tracking
        self.is_initialized = False
        self.last_valid_time = 0.0
        self.last_valid_bis = 98.0
        self.last_valid_map = 90.0
        self.last_valid_hr = 75.0
        
        self.consecutive_noise_sec = 0.0
        self.sensor_hold_active = False

    def process(
        self,
        time_sec: float,
        raw_bis: float,
        raw_map: float,
        raw_hr: float,
        sqi_percent: float = 100.0,
        emg_db: float = 25.0,
        dt_seconds: float = 1.0
    ) -> FilteredTelemetry:
        """
        Processes raw incoming telemetry and outputs validated/cleaned signals.
        """
        cfg = self.config

        # Initialize on first frame
        if not self.is_initialized:
            self.is_initialized = True
            self.last_valid_time = time_sec
            self.last_valid_bis = raw_bis
            self.last_valid_map = raw_map
            self.last_valid_hr = raw_hr

        is_artifact = False
        reasons = []

        # ----------------------------------------------------
        # 1. Check Signal Quality Index (SQI) and Spectral EMG
        # ----------------------------------------------------
        if sqi_percent < cfg.min_sqi_threshold:
            is_artifact = True
            reasons.append(f"Low SQI ({sqi_percent:.0f}%)")

        if emg_db > cfg.max_emg_threshold_db:
            is_artifact = True
            reasons.append(f"High EMG Noise ({emg_db:.1f} dB)")

        # ----------------------------------------------------
        # 2. Dynamic Velocity Gate Check
        # ----------------------------------------------------
        dt = max(dt_seconds, 1e-4)
        bis_rate_of_change = abs(raw_bis - self.last_valid_bis) / dt
        if bis_rate_of_change > cfg.velocity_threshold_bis_per_sec:
            is_artifact = True
            reasons.append(f"Velocity Breach ({bis_rate_of_change:.1f} units/s)")

        # ----------------------------------------------------
        # 3. Sustained Noise & "Sensor Hold Mode" Failsafe
        # ----------------------------------------------------
        if is_artifact:
            self.consecutive_noise_sec += dt_seconds
            if self.consecutive_noise_sec >= cfg.hold_trigger_duration_seconds:
                self.sensor_hold_active = True
                status_msg = f"[ELECTROCAUTERY NOISE — SENSOR HOLD ACTIVE ({self.consecutive_noise_sec:.0f}s)]"
            else:
                status_msg = f"[ARTIFACT REJECTED: {', '.join(reasons)}]"
            
            # During artifact, hold last valid clean physiological values
            filtered_bis = self.last_valid_bis
            filtered_map = self.last_valid_map
            filtered_hr = self.last_valid_hr
        else:
            # Clean signal restored
            self.consecutive_noise_sec = 0.0
            self.sensor_hold_active = False
            status_msg = "[SIGNAL CLEAN]"

            # Update rolling median window
            self.bis_buffer.append(raw_bis)
            self.map_buffer.append(raw_map)
            self.hr_buffer.append(raw_hr)

            # Max buffer length based on median window
            max_len = max(3, int(cfg.median_window_seconds / dt))
            while len(self.bis_buffer) > max_len:
                self.bis_buffer.popleft()
                self.map_buffer.popleft()
                self.hr_buffer.popleft()

            # Compute rolling median
            filtered_bis = float(np.median(self.bis_buffer))
            filtered_map = float(np.median(self.map_buffer))
            filtered_hr = float(np.median(self.hr_buffer))

            # Update last valid states
            self.last_valid_time = time_sec
            self.last_valid_bis = filtered_bis
            self.last_valid_map = filtered_map
            self.last_valid_hr = filtered_hr

        return FilteredTelemetry(
            time_seconds=time_sec,
            raw_bis=raw_bis,
            filtered_bis=filtered_bis,
            raw_map=raw_map,
            filtered_map=filtered_map,
            raw_hr=raw_hr,
            filtered_hr=filtered_hr,
            sqi_percent=sqi_percent,
            emg_db=emg_db,
            is_artifact=is_artifact,
            sensor_hold_active=self.sensor_hold_active,
            status_message=status_msg
        )

    def reset(self) -> None:
        self.bis_buffer.clear()
        self.map_buffer.clear()
        self.hr_buffer.clear()
        self.is_initialized = False
        self.last_valid_time = 0.0
        self.last_valid_bis = 98.0
        self.last_valid_map = 90.0
        self.last_valid_hr = 75.0
        self.consecutive_noise_sec = 0.0
        self.sensor_hold_active = False
