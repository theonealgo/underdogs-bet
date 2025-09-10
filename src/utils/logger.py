import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "mlb_prediction_system",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a comprehensive logger for the MLB prediction system
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = "mlb_prediction_system") -> logging.Logger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger

def configure_logging_for_app():
    """
    Configure logging for the entire MLB prediction application
    """
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up main application logger
    app_logger = setup_logger(
        name="mlb_prediction_system",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.path.join(log_dir, "mlb_prediction.log")
    )
    
    # Set up component-specific loggers
    component_loggers = [
        "data_collectors.baseball_savant_scraper",
        "data_collectors.oddshark_scraper", 
        "data_storage.database",
        "models.prediction_models",
        "features.feature_engineering",
        "utils.scheduler",
        "api.prediction_api",
        "backtesting.backtester"
    ]
    
    for component in component_loggers:
        setup_logger(
            name=component,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.path.join(log_dir, f"{component.split('.')[-1]}.log")
        )
    
    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)
    
    # Suppress excessive logging from some libraries
    logging.getLogger("pybaseball").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    
    app_logger.info("Logging configuration completed")
    return app_logger

class LoggingContext:
    """
    Context manager for temporary logging configuration
    """
    
    def __init__(self, logger_name: str, level: str):
        self.logger_name = logger_name
        self.level = level
        self.original_level = None
        
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(getattr(logging, self.level.upper()))
        return logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        if self.original_level is not None:
            logger.setLevel(self.original_level)

def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func.__name__} executed successfully in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator

def log_data_operation(logger: logging.Logger, operation: str):
    """
    Decorator to log data operations with details
    
    Args:
        logger: Logger instance to use
        operation: Description of the operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {operation}")
            try:
                result = func(*args, **kwargs)
                
                # Log result details if it's a DataFrame or list
                if hasattr(result, '__len__'):
                    if hasattr(result, 'shape'):  # DataFrame
                        logger.info(f"{operation} completed: {result.shape[0]} rows, {result.shape[1]} columns")
                    else:  # List or similar
                        logger.info(f"{operation} completed: {len(result)} items")
                else:
                    logger.info(f"{operation} completed successfully")
                
                return result
            except Exception as e:
                logger.error(f"{operation} failed: {str(e)}")
                raise
        return wrapper
    return decorator

# Initialize logging when module is imported
if __name__ != "__main__":
    configure_logging_for_app()
