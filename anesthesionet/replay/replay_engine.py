"""
Mode B Historical Replay Streaming Engine
Streams high-resolution intraoperative cases line-by-line from VitalDB.
"""

from typing import Iterator
import time
from .vital_case import VitalDBCase, IntraoperativeFrame, generate_benchmark_vitaldb_case


class VitalDBReplayEngine:
    """
    Mode B Real-World Historical Replay Engine.
    Simulates a live patient telemetry feed using retrospective surgical recordings.
    """

    def __init__(self, case: VitalDBCase | None = None):
        self.case = case or generate_benchmark_vitaldb_case()
        self.current_index = 0
        self.total_frames = len(self.case)

    def step(self) -> tuple[IntraoperativeFrame, bool]:
        """
        Advances the replay by one second frame.
        Returns:
            (frame, is_finished)
        """
        if self.current_index >= self.total_frames:
            # Reached end of case recording
            last_frame = self.case.get_frame(self.total_frames - 1)
            return last_frame, True

        frame = self.case.get_frame(self.current_index)
        self.current_index += 1
        is_finished = (self.current_index >= self.total_frames)

        return frame, is_finished

    def stream(self, time_scale: float = 1.0) -> Iterator[IntraoperativeFrame]:
        """
        Generator yielding frames with real-time or time-scaled pacing.
        time_scale: 1.0 = real-time, 10.0 = 10x speed, 0.0 = unthrottled maximum speed.
        """
        self.reset()
        while self.current_index < self.total_frames:
            t0 = time.perf_counter()
            frame, is_finished = self.step()
            yield frame

            if is_finished:
                break

            if time_scale > 0.0:
                target_delay = 1.0 / time_scale
                elapsed = time.perf_counter() - t0
                remaining = target_delay - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)

    def reset(self) -> None:
        """Resets the playback position to beginning of case."""
        self.current_index = 0

    @property
    def progress_percent(self) -> float:
        if self.total_frames == 0:
            return 100.0
        return (self.current_index / self.total_frames) * 100.0
