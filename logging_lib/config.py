import logging
import logging.handlers
from pathlib import Path
from typing import Optional

import yaml


class LoggingConfig:
    """Configuration class for logging settings"""
    
    def __init__(
        self,
        level: int = logging.INFO,
        log_dir: str = "logs",
        log_file: str = "app.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: Optional[str] = None,
        quiet_loggers: Optional[list] = None,
        console_enabled: bool = True,
    ):
        self.level = level
        self.log_dir = log_dir
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.format_string = (
            format_string
            or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.quiet_loggers: list = quiet_loggers or []
        self.console_enabled = console_enabled


def get_default_logging_config() -> LoggingConfig:
    """Load logging defaults from logging_lib/default_config.yaml."""
    config_path = Path(__file__).with_name("default_config.yaml")
    if not config_path.exists():
        return LoggingConfig()

    with config_path.open("r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file) or {}

    logging_yaml = yaml_config.get("logging", {})
    level_name = str(logging_yaml.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    return LoggingConfig(
        level=level,
        log_dir=logging_yaml.get("log_dir", "logs"),
        log_file=logging_yaml.get("log_file", "app.log"),
        max_bytes=logging_yaml.get("max_bytes", 10 * 1024 * 1024),
        backup_count=logging_yaml.get("backup_count", 5),
        format_string=logging_yaml.get("format_string"),
        quiet_loggers=logging_yaml.get("quiet_loggers", []),
        console_enabled=bool(logging_yaml.get("console_enabled", True)),
    )


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    
    Args:
        config: LoggingConfig object. If None, uses default configuration.
    
    Returns:
        Configured logger instance
    """
    if config is None:
        config = get_default_logging_config()
    
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(config.level)
    logger.handlers.clear()

    formatter = logging.Formatter(config.format_string)
    if config.console_enabled:
        logger.addHandler(_build_console_handler(config, formatter))
    logger.addHandler(_build_file_handler(config, formatter, log_path))

    _quiet_third_party_loggers(config.quiet_loggers)

    return logger


def _build_console_handler(
    config: LoggingConfig,
    formatter: logging.Formatter,
) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(config.level)
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(
    config: LoggingConfig,
    formatter: logging.Formatter,
    log_path: Path,
) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        log_path / config.log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
    )
    handler.setLevel(config.level)
    handler.setFormatter(formatter)
    return handler


def _quiet_third_party_loggers(names: list[str]) -> None:
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
