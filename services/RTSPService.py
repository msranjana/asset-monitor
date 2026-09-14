"""Reads an RTSP stream or local video file and keeps the newest frame ready."""

import os
import threading
import time

import cv2

from console import say


def _is_network_stream(source: str) -> bool:
    lower = source.lower()
    return lower.startswith(("rtsp://", "rtmp://", "http://", "https://"))

# OpenCV reads RTSP through FFmpeg, which uses UDP by default. UDP loses packets on a
# busy network and FFmpeg then floods the console with h264 decode errors, so ask for
# TCP instead and keep FFmpeg's own logging quiet. setdefault means a value already
# set in the environment still wins.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


class RTSPService:
    """Opens an RTSP url and keeps reading frames in its own thread.

    Detections do not read from the camera themselves, they ask this service for the
    latest frame with `read()`. That way one camera connection can feed many detections.
    """

    def __init__(self, rtsp_url, reconnect_delay=5, loop_video=True):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.loop_video = loop_video
        self._is_video_file = not _is_network_stream(rtsp_url) and os.path.isfile(rtsp_url)
        self._file_frame_interval = 1.0 / 25.0

        self._capture = None
        self._latest_frame = None
        # Counter so a detection can tell a new frame from an already processed one.
        self._frame_number = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, name="rtsp-reader", daemon=True)
        self._thread.start()
        kind = "video file" if self._is_video_file else "stream"
        say(f"[RTSPService] started for {kind}: {self.rtsp_url}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release_capture()
        say("[RTSPService] stopped")

    def read(self):
        """Return (frame_number, frame) of the newest frame, or (0, None) if nothing yet."""
        with self._lock:
            if self._latest_frame is None:
                return 0, None
            return self._frame_number, self._latest_frame.copy()

    def is_connected(self):
        return self._capture is not None and self._capture.isOpened()

    def _read_loop(self):
        while not self._stop_event.is_set():
            if not self.is_connected() and not self._connect():
                self._stop_event.wait(self.reconnect_delay)
                continue

            success, frame = self._capture.read()
            if not success or frame is None:
                if self._is_video_file and self.loop_video and self._capture is not None:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                say("[RTSPService] lost the stream, reconnecting...")
                self._release_capture()
                self._stop_event.wait(self.reconnect_delay)
                continue

            with self._lock:
                self._latest_frame = frame
                self._frame_number += 1

            if self._is_video_file:
                self._stop_event.wait(self._file_frame_interval)

    def _connect(self):
        say(f"[RTSPService] connecting to {self.rtsp_url}")
        capture = cv2.VideoCapture(self.rtsp_url)
        if not capture.isOpened():
            capture.release()
            say("[RTSPService] connection failed")
            return False

        # Keep the internal buffer tiny so we always work on a fresh frame.
        if not self._is_video_file:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if self._is_video_file and fps and fps > 1:
            self._file_frame_interval = 1.0 / fps
        self._capture = capture
        say("[RTSPService] connected")
        return True

    def _release_capture(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
