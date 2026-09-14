"""Handles alerts raised by detections. Email sending comes later, once SMTP is ready."""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from console import say


class AlertType:
    """The three severities a detection can raise."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    source: str
    alert_type: str
    message: str
    image_base64: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class AlertEngine:
    """Consumes the alert queue in its own thread and delivers each alert.

    Right now delivery only prints the alert. When the SMTP credentials are added to
    .env, `_send_email` is the only method that needs to be filled in.
    """

    def __init__(self, smtp_config=None):
        self.smtp_config = smtp_config or {}

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._deliver_loop, name="alert-engine", daemon=True)
        self._thread.start()
        say("[AlertEngine] started")

    def stop(self):
        self._queue.join()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        say("[AlertEngine] stopped")

    def push(self, source, alert_type, message, image_base64=None):
        self._queue.put(
            Alert(
                source=source,
                alert_type=alert_type,
                message=message,
                image_base64=image_base64,
            )
        )

    def _deliver_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._deliver(alert)
            except Exception as error:
                say(f"[AlertEngine] could not deliver alert: {error}")
            finally:
                self._queue.task_done()

    def _deliver(self, alert):
        has_image = "yes" if alert.image_base64 else "no"
        say(
            f"[AlertEngine] {alert.alert_type} from {alert.source}: "
            f"{alert.message} (image attached: {has_image})"
        )
        self._send_email(alert)

    def _send_email(self, alert):
        if not self.smtp_config.get("host"):
            # No credentials yet, so nothing to send.
            return

        # TODO: build the message and send it once the SMTP account is available.
        raise NotImplementedError("SMTP sending is not implemented yet")
