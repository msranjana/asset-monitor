"""Entry point: reads the camera once and feeds every attached detection in parallel.

To add a detection:
  1. create a class in detectors/ that extends BaseDetector
  2. add one DetectionWrapper line to `build_detections()` below
"""

import time

import config
from console import say
from detectors.asset_monitoring_detector import AssetMonitoringDetector, AssetROI
from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from engines.alert_engine import AlertEngine
from engines.event_engine import EventEngine
from services.DetectionWrapper import DetectionWrapper
from services.RTSPService import RTSPService


def build_detections():
    """Return the detections to run. DetectionWrapper(name, fps, detector)."""
    # Fixed ROI per camera; baseline comes from the live stream after warmup
    # (see AssetMonitoringDetector.baseline_warmup_frames). Optional file:
    #   AssetROI(..., reference_image_path="assets/bench_12_reference.jpg")
    # Example:
    #   AssetROI(name="bench_12", bbox=(400, 120, 620, 340)),
    assets = []
    detections = [
        DetectionWrapper("smoke_and_fire_detection", 2, SmokeAndFireDetector()),
    ]
    if assets:
        detections.append(
            DetectionWrapper("asset_monitoring", 4, AssetMonitoringDetector(assets))
        )
    return detections


def main():
    frame_source = config.VIDEO_PATH or config.RTSP_URL
    if not frame_source:
        say("Set RTSP_URL or VIDEO_PATH in the .env file")
        return

    alert_engine = AlertEngine(
        smtp_config={
            "host": config.SMTP_HOST,
            "port": config.SMTP_PORT,
            "username": config.SMTP_USERNAME,
            "password": config.SMTP_PASSWORD,
            "from_email": config.ALERT_FROM_EMAIL,
            "to_emails": config.ALERT_TO_EMAILS,
        }
    )
    event_engine = EventEngine(config.LOG_FILE_PATH, alert_engine=alert_engine)
    reader = RTSPService(frame_source, reconnect_delay=config.RECONNECT_DELAY)

    detections = build_detections()
    if not detections:
        say("No detection attached yet, add one in build_detections() in main.py")
        return

    alert_engine.start()
    event_engine.start()
    reader.start()

    # Each wrapper runs in its own thread, so all detections work at the same time.
    for detection in detections:
        detection.start(reader=reader, event_engine=event_engine)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        say("\nshutting down...")
    finally:
        for detection in detections:
            detection.stop()
        reader.stop()
        event_engine.stop()
        alert_engine.stop()


if __name__ == "__main__":
    main()
