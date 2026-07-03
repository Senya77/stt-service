"""Конфиг логгера для приложения."""

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(levelname)s \t %(asctime)s - %(name)s - %(message)s",
            "datefmt": "%d/%m/%Y %H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
    },
}
