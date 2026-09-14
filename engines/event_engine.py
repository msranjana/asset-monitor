"""Collects logs and alerts from every detection and writes them to the log file."""

import os
import queue
import threading
from datetime import datetime

from console import say


class EventEngine:
    """Single place where all detections send their events.

    Detectors never write to files themselves, they hand the event over here. The
    engine writes every event as one line in the .log file, and when the event is an
    alert it also drops it into the alert engine's queue.
    """

    def __init__(self, log_file_path, alert_engine=None):
        self.log_file_path = log_file_path
        self.alert_engine = alert_engine

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        folder = os.path.dirname(self.log_file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._write_loop, name="event-engine", daemon=True)
        self._thread.start()
        say(f"[EventEngine] writing logs to {self.log_file_path}")

    def stop(self):
        # Let the queue drain first so no event is lost on shutdown.
        self._queue.join()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        say("[EventEngine] stopped")

    def log(self, source, message, level="INFO"):
        """Write a plain log line, for example "no smoke found"."""
        self._queue.put(self._build_line(level, source, message))

    def alert(self, source, alert_type, message, image_base64=None):
        """Write a log line and push the alert into the alert queue."""
        self._queue.put(self._build_line(f"ALERT/{alert_type}", source, message))

        if self.alert_engine:
            self.alert_engine.push(
                source=source,
                alert_type=alert_type,
                message=message,
                image_base64=image_base64,
            )

    def _build_line(self, level, source, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} | {level} | {source} | {message}"

    def _write_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(line + "\n")
                say(line)
            finally:
                self._queue.task_done()
