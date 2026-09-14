"""All settings are read from the .env file so nothing is hardcoded in the code."""

import os

from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "")
# Optional local file for testing (mp4, avi, …). When set, overrides RTSP_URL.
VIDEO_PATH = os.getenv("VIDEO_PATH", "")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/events.log")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

SMOKE_FIRE_MODEL_PATH = os.getenv("SMOKE_FIRE_MODEL_PATH", "models/fire_smoke_yolov8n.pt")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")

# Comma separated list in .env -> list of addresses here
ALERT_TO_EMAILS = [
    email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()
]
