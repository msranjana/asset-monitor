"""Runs one detection in its own thread at its own frames-per-second."""

import threading
import time

from console import say


class DetectionWrapper:
    """Feeds frames from the RTSP service into one detector.

    Example: DetectionWrapper("smoke_detection", 10, SmokeDetector())
    -> gives the smoke detector up to 10 frames per second, in a separate thread,
    so every detection you attach in main.py runs in parallel.
    """

    def __init__(self, name, fps, detector):
        self.name = name
        self.fps = fps
        self.detector = detector

        self._reader = None
        self._event_engine = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, reader, event_engine):
        self._reader = reader
        self._event_engine = event_engine
        # The detector gets the engine so it can log and raise alerts by itself.
        self.detector.attach(self.name, event_engine)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=self.name, daemon=True)
        self._thread.start()
        say(f"[{self.name}] started at {self.fps} fps")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        say(f"[{self.name}] stopped")

    def _run_loop(self):
        frame_interval = 1.0 / self.fps
        last_frame_number = 0

        while not self._stop_event.is_set():
            started_at = time.time()
            frame_number, frame = self._reader.read()

            # Skip if there is no frame yet, or if it is the one we already handled.
            if frame is not None and frame_number != last_frame_number:
                last_frame_number = frame_number
                try:
                    self.detector.process(frame)
                except Exception as error:
                    self._event_engine.log(self.name, f"detector failed: {error}", level="ERROR")

            # Wait out the rest of this frame's time slot.
            time_left = frame_interval - (time.time() - started_at)
            if time_left > 0:
                self._stop_event.wait(time_left)
