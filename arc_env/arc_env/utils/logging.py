import logging
import sys
from typing import Optional, Union, Dict, Any
from pathlib import Path

# Standard log levels
LOG_LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache for already configured loggers to avoid duplicate handlers
_configured_loggers: Dict[str, logging.Logger] = {}

def setup_logger(
    logger_name: str = "arc_env",
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    propagate: bool = False, # Whether to propagate to parent loggers (e.g., root)
    use_console_handler: bool = True,
    use_file_handler: bool = True # Only if log_file is provided
) -> logging.Logger:
    """
    Sets up and configures a logger instance.

    Args:
        logger_name: The name for the logger.
        level: Logging level (e.g., "INFO", "DEBUG", or logging.INFO).
        log_file: Optional path to a file where logs should be saved.
        log_format: The format string for log messages.
        date_format: The format string for dates in log messages.
        propagate: If True, messages are passed to handlers of ancestor loggers.
        use_console_handler: Whether to add a handler for console output (stderr).
        use_file_handler: Whether to add a handler for file output (if log_file is specified).

    Returns:
        A configured logging.Logger instance.
    """
    if logger_name in _configured_loggers:
        # Logger already configured, possibly return it or ensure its level is updated
        # For simplicity, return the existing one. If config needs to change, it's more complex.
        # To allow reconfiguration, one might clear existing handlers first.
        # print(f"Logger '{logger_name}' already configured. Returning existing instance.")
        return _configured_loggers[logger_name]

    logger = logging.getLogger(logger_name)

    # Determine numeric log level
    if isinstance(level, str):
        numeric_level = LOG_LEVELS.get(level.upper())
        if numeric_level is None:
            print(f"Warning: Invalid log level string '{level}'. Defaulting to INFO.")
            numeric_level = logging.INFO
    elif isinstance(level, int):
        numeric_level = level
    else:
        print(f"Warning: Invalid log level type '{type(level)}'. Defaulting to INFO.")
        numeric_level = logging.INFO

    logger.setLevel(numeric_level)
    logger.propagate = propagate

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console Handler (stderr)
    if use_console_handler:
        # Check if a similar console handler already exists to avoid duplicates from multiple calls
        has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in logger.handlers)
        if not has_console_handler:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            # console_handler.setLevel(numeric_level) # Handler level can be different from logger level
            logger.addHandler(console_handler)

    # File Handler
    if log_file and use_file_handler:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

        # Check for existing file handler for this specific file path
        has_file_handler_for_path = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file_path for h in logger.handlers)
        if not has_file_handler_for_path:
            file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            # file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)

    _configured_loggers[logger_name] = logger
    return logger


def get_logger(logger_name: str = "arc_env") -> logging.Logger:
    """
    Retrieves a logger instance. If not already configured by `setup_logger`,
    it might be a basic logger from `logging.getLogger`.
    It's recommended to call `setup_logger` once for each main logger name
    at the beginning of the application or module import.
    """
    # This function primarily serves as a convenient way to get a logger
    # that is expected to have been set up.
    # If it's not in _configured_loggers, it means setup_logger wasn't called for it,
    # or it's a child logger that will propagate to a configured parent.
    return logging.getLogger(logger_name)


# Example Usage:
if __name__ == "__main__":
    # Setup a main application logger
    main_logger = setup_logger(
        logger_name="MyARCApp",
        level="DEBUG", # Log everything from DEBUG upwards
        log_file="my_arc_app.log"
    )
    main_logger.debug("This is a debug message for MyARCApp.")
    main_logger.info("This is an info message for MyARCApp.")
    main_logger.warning("This is a warning from MyARCApp.")

    # Get the same logger instance again (it's cached by name)
    another_ref_main_logger = get_logger("MyARCApp")
    another_ref_main_logger.error("Error message from another reference to MyARCApp logger.")

    # Setup a specific module logger (e.g., for a solver)
    # It can have its own settings or propagate to "MyARCApp" if "MyARCApp.Solver"
    # and propagate=True (and MyARCApp is configured).
    # If logger_name is "Solver", it's independent unless root logger is configured.
    solver_logger = setup_logger(
        logger_name="SolverModule",
        level="INFO",
        log_file="solver_module.log", # Separate log file
        use_console_handler=True # Solver can also log to console
    )
    solver_logger.info("SolverModule initialized successfully.")
    solver_logger.debug("This solver debug message won't appear if level is INFO.") # Not shown

    # Example of a child logger that propagates if parent is configured
    # If "MyARCApp" is set up, "MyARCApp.submodule" will use its handlers by default
    # if propagate=True for "MyARCApp.submodule" (which is default for getLogger).
    # setup_logger sets propagate=False by default for the named logger itself.
    # Child loggers get their own instance.

    # If we want child loggers to use parent's handlers, parent should have propagate=True (less common for named loggers)
    # OR child loggers just work via propagation if they don't have their own handlers and propagate=True.
    # logging.getLogger automatically creates parent-child relationship by name.
    child_logger = get_logger("MyARCApp.utils") # Gets a child of MyARCApp
    # If MyARCApp was set up with its own handlers and MyARCApp.utils doesn't have its own,
    # messages to child_logger will go to MyARCApp's handlers if child_logger.propagate is True.
    # By default, getLogger makes new loggers with propagate=True.
    # setup_logger sets propagate to False for the logger it configures, to avoid duplicate logs
    # if root logger also has handlers.

    child_logger.info("Info message from MyARCApp.utils (should go to MyARCApp's handlers if MyARCApp allows propagation or if child has no handlers of its own).")
    # The behavior of propagation can be complex. Generally:
    # - setup_logger(name, propagate=False) makes 'name' not send to ancestors.
    # - get_logger(name.child) will propagate to 'name' if 'name.child' has no handlers.

    print("\nCheck 'my_arc_app.log' and 'solver_module.log' for output.")
    # Clean up dummy log files for example
    # import os
    # if os.path.exists("my_arc_app.log"): os.remove("my_arc_app.log")
    # if os.path.exists("solver_module.log"): os.remove("solver_module.log")
