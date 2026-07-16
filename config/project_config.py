"""
Project Configuration Module
Centralizes configuration for all libraries and services.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from logging_lib import LoggingConfig, get_default_logging_config, get_logger, setup_logging
from llm_api_lib import LLMClient


class ProjectConfig:
    """Central project configuration manager"""
    
    def __init__(self):
        self.logger: Optional[logging.Logger] = None
        self.llm_client: Optional[LLMClient] = None
        self.project_overrides = self._load_yaml_file(
            Path(__file__).with_name("project_config.yaml")
        )

    def _load_yaml_file(self, file_path: Path) -> dict:
        """Load YAML content from a file path, returning empty dict if missing."""
        if not file_path.exists():
            return {}
        with file_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _logging_level_from_name(self, level_name: str) -> int:
        """Convert logging level name to logging module constant."""
        level = getattr(logging, level_name.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid logging level in YAML: {level_name}")
        return level

    def _build_logging_config(
        self,
        level: Optional[int],
        log_dir: Optional[str],
        log_file: Optional[str],
        max_bytes: Optional[int],
        backup_count: Optional[int],
        console_enabled: Optional[bool],
    ) -> LoggingConfig:
        """Build final logging config from library defaults + project overrides + runtime args."""
        project_logging = self.project_overrides.get("logging", {})
        base = get_default_logging_config()

        resolved_level = level
        if resolved_level is None:
            override_level = project_logging.get("level")
            if override_level is not None:
                resolved_level = self._logging_level_from_name(str(override_level))
            else:
                resolved_level = base.level

        return LoggingConfig(
            level=resolved_level,
            log_dir=log_dir or project_logging.get("log_dir") or base.log_dir,
            log_file=log_file or project_logging.get("log_file") or base.log_file,
            max_bytes=max_bytes or project_logging.get("max_bytes") or base.max_bytes,
            backup_count=backup_count or project_logging.get("backup_count") or base.backup_count,
            format_string=project_logging.get("format_string") or base.format_string,
            quiet_loggers=project_logging.get("quiet_loggers") or base.quiet_loggers,
            console_enabled=(
                console_enabled
                if console_enabled is not None
                else project_logging.get("console_enabled", base.console_enabled)
            ),
        )

    def _resolve_llm_value(self, section: dict, runtime_value, key: str):
        """Pick runtime value first, then project override, else None."""
        if runtime_value is not None:
            return runtime_value
        return section.get(key)

    def _resolve_llm_options(
        self,
        provider: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        temperature: Optional[float],
        use_env: Optional[bool],
    ) -> dict:
        section = self.project_overrides.get("llm", {})
        resolved_use_env = self._resolve_llm_value(section, use_env, "use_env")
        return {
            "provider": self._resolve_llm_value(section, provider, "provider"),
            "model": self._resolve_llm_value(section, model, "model"),
            "base_url": self._resolve_llm_value(section, base_url, "base_url"),
            "temperature": self._resolve_llm_value(
                section, temperature, "temperature"
            ),
            "load_env": True if resolved_use_env is None else resolved_use_env,
        }

    def _log_llm_configuration(self, source: str) -> None:
        config = self.llm_client.config
        self.logger.info("[OK] LLM configured %s", source)
        self.logger.info(
            "  LLM params: provider=%s, model=%s, base_url=%s, temperature=%s",
            config.provider,
            config.model,
            config.base_url,
            config.temperature,
        )
        
    def configure_logging(
        self,
        level: Optional[int] = None,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
        console_enabled: Optional[bool] = None,
    ) -> logging.Logger:
        """
        Configure logging system.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_file: Log filename
            max_bytes: Max file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger instance
        """
        logging_cfg = self._build_logging_config(
            level=level,
            log_dir=log_dir,
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
            console_enabled=console_enabled,
        )
        self.logger = setup_logging(logging_cfg)
        
        # Log successful configuration
        self.logger.info("[OK] Logging configured")
        self.logger.info(
            "  Logging params: "
            f"level={logging.getLevelName(logging_cfg.level)}, "
            f"log_dir={logging_cfg.log_dir}, "
            f"log_file={logging_cfg.log_file}, "
            f"max_bytes={logging_cfg.max_bytes}, "
            f"backup_count={logging_cfg.backup_count}"
        )
        
        return self.logger
    
    def configure_llm(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        use_env: Optional[bool] = None,
    ) -> LLMClient:
        """
        Configure LLM client.
        
        Args:
            provider: LLM provider (local, openai, claude, gemini)
            model: Model name
            base_url: Base URL for the provider
            temperature: Temperature parameter
            use_env: Whether to load from environment variables
            
        Returns:
            Configured LLM client instance
        """
        if not self.logger:
            self.logger = get_logger(__name__)

        options = self._resolve_llm_options(
            provider, model, base_url, temperature, use_env
        )
        try:
            self.llm_client = LLMClient(**options)
            source = "from environment" if options["load_env"] else "with YAML/custom parameters"
            self._log_llm_configuration(source)
        except Exception as exc:
            self.logger.error("[ERROR] Failed to configure LLM: %s", exc)
            raise
        
        return self.llm_client
    
    def configure_all(
        self,
        # Logging config
        log_level: Optional[int] = None,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_max_bytes: Optional[int] = None,
        log_backup_count: Optional[int] = None,
        log_console_enabled: Optional[bool] = None,
        # LLM config
        llm_use_env: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_temperature: Optional[float] = None,
    ) -> dict:
        """
        Configure all project components in one call.
        
        Args:
            log_level: Logging level
            log_dir: Logs directory
            log_file: Log filename
            log_max_bytes: Max log file size
            log_backup_count: Log backup count
            llm_use_env: Load LLM from environment
            llm_provider: LLM provider
            llm_model: LLM model
            llm_base_url: LLM base URL
            llm_temperature: LLM temperature
            
        Returns:
            Dictionary with configured components
        """
        # Configure logging first
        self.configure_logging(
            level=log_level,
            log_dir=log_dir,
            log_file=log_file,
            max_bytes=log_max_bytes,
            backup_count=log_backup_count,
            console_enabled=log_console_enabled,
        )
        
        # Configure LLM
        self.configure_llm(
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
            temperature=llm_temperature,
            use_env=llm_use_env,
        )
        
        self.logger.info("=" * 50)
        self.logger.info("[OK] All configurations loaded successfully")
        self.logger.info("=" * 50)
        
        return {
            "logger": self.logger,
            "llm_client": self.llm_client,
        }


# Singleton instance
_project_config: Optional[ProjectConfig] = None


def get_project_config() -> ProjectConfig:
    """Get or create project config singleton"""
    global _project_config
    if _project_config is None:
        _project_config = ProjectConfig()
    return _project_config


def configure_project(**kwargs) -> dict:
    """
    Configure all project components.
    
    Usage:
        config = configure_project()
        logger = config["logger"]
        llm_client = config["llm_client"]

    Notes:
                - Library defaults are owned by each library:
                    - logging_lib/default_config.yaml
                    - llm_api_lib/default_config.yaml
                - This module only applies project overrides from config/project_config.yaml
        - Any kwargs provided here override all YAML values
    """
    project_cfg = get_project_config()
    return project_cfg.configure_all(**kwargs)
