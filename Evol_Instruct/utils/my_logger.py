import logging
import sys
logging.getLogger().handlers = []
def setup_logger(logger_name, log_file=None, log_level=logging.DEBUG):
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Check if logger already has handlers to avoid adding them multiple times
    if not logger.handlers:
        # Create a handler that logs to standard output if no log file is provided
        if log_file:
            handler = logging.FileHandler(log_file)  # Logs to the provided file
        else:
            handler = logging.StreamHandler(sys.stdout)  # Logs to standard output

        # Set the log level for the handler
        handler.setLevel(log_level)

        # Create a formatter with desired format
        log_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s')
        
        # Add the formatter to the handler
        handler.setFormatter(log_format)

        # Add the handler to the logger
        logger.addHandler(handler)

    
    return logger

# Example of using the logger
# logger = setup_logger('agent')  # Logs to standard output by default
logger = setup_logger(__name__)
# You can also pass a file name to log to a file instead
# logger = setup_logger("output.log")  # Logs to output.log file

# Log some messages with different severity levels
if __name__ == "__main__":
    logger = setup_logger('agent')  # Logs to standard output by default
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
