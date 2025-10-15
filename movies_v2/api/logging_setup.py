import logging, json, os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_dict"):
            # ensure JSON-serializable
            try:
                json.dumps(record.extra_dict)
                base.update(record.extra_dict)
            except Exception:
                base["bad_extra"] = str(record.extra_dict)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = TimedRotatingFileHandler(LOG_PATH, when="D", backupCount=14, encoding="utf-8")
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.propagate = False
    return logger
