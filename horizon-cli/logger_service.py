import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

LOG_JSON = False
LOGGER_NAME = "horizon"


class JSONFormatter(logging.Formatter):
    RESERVED = {"args", "msg", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process"}
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "time": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        event = getattr(record, "event", None)
        if event:
            base["event"] = event
        detail = getattr(record, "detail", None)
        if detail and detail != base.get("message"):
            base["detail"] = detail
        # Merge any structured extras if present
        for k, v in record.__dict__.items():
            if k not in base and k not in self.RESERVED and k not in {"event", "detail"}:
                base[k] = v
        return json.dumps(base, ensure_ascii=False)


def init_logger(log_json: bool = False, log_level: str = "INFO"):
    global LOG_JSON
    try:
        from typer.models import OptionInfo
        if isinstance(log_json, OptionInfo):
            log_json = False
        if isinstance(log_level, OptionInfo):
            log_level = "INFO"
    except Exception:  # pragma: no cover - defensive
        pass
    LOG_JSON = log_json
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    # Clear existing handlers (for repeated test runs)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if log_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger.debug("Logger initialized", extra={"event": "logger_initialized", "mode": "json" if log_json else "text"})
    return logger


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, **fields):
    logger = get_logger()
    msg = fields.pop("message", event)
    # Avoid reserved overwrite: move original message into detail if different
    extra = {"event": event}
    if fields:
        # rename any accidental 'message' key already popped
        if 'detail' not in fields and msg != event:
            extra['detail'] = msg
        extra.update(fields)
    logger.info(msg, extra=extra)
