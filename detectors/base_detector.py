"""Base class every detection must extend."""

import base64
from abc import ABC, abstractmethod

import cv2

from engines.alert_engine import AlertType


class BaseDetector(ABC):
    """Write a new detection by extending this class and filling in `process`.

    Inside `process` you get one frame and you can call:
        self.log("message")                                   -> line in the log file
        self.alert(image_base64, AlertType.CRITICAL, "text")  -> log line + alert queue
        self.to_base64(frame)                                 -> frame as a base64 string
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self._event_engine = None

    def attach(self, name, event_engine):
        """Called by DetectionWrapper before the detection starts running."""
        self.name = name
        self._event_engine = event_engine
        self.on_start()

    def on_start(self):
        """Optional hook for one time setup, for example loading a model."""

    @abstractmethod
    def process(self, frame):
        """Look at a single frame and log or alert when something is found."""

    def log(self, message, level="INFO"):
        if self._event_engine:
            self._event_engine.log(self.name, message, level=level)

    def alert(self, image_base64, alert_type=AlertType.NORMAL, message=""):
        if self._event_engine:
            self._event_engine.alert(
                source=self.name,
                alert_type=alert_type,
                message=message,
                image_base64=image_base64,
            )

    def to_base64(self, frame, quality=80):
        """Turn a frame into a base64 JPEG string, ready to attach to an alert."""
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return None
        return base64.b64encode(buffer).decode("utf-8")
