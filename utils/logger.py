import logging
import os
from typing import Optional

class ContextLogger:
    def __init__(self, logger):
        self.logger = logger
        
    def __enter__(self):
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Context failed: {exc_val}", exc_info=True)

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Enhanced logger with context support and symbols
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Symbol mappings
    LEVEL_SYMBOLS = {
        logging.DEBUG: '🐛 ',
        logging.INFO: 'ℹ️ ',
        logging.WARNING: '⚠️ ',
        logging.ERROR: '❌ ',
        logging.CRITICAL: '💥 '
    }

    class SymbolFormatter(logging.Formatter):
        def format(self, record):
            record.msg = f"{LEVEL_SYMBOLS.get(record.levelno, '')}{record.msg}"
            return super().format(record)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = SymbolFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add context method
    def context(self, message):
        self.info(f"Starting: {message}")
        return ContextLogger(self)
        
    logger.context = context.__get__(logger)
    
    return logger