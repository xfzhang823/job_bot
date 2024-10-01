import logging
import logging.handlers
import os
from utils.generic_utils import find_project_root

# Ensure logs directory exists
root_dir = find_project_root()
logs_dir = os.path.join(root_dir, "logs")

# Ensure the logs directory exists
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    logging.info(f"Created logs directory: {logs_dir}")

# Set up log file rotation: max 100MB per file, up to 5 backup files
log_file_path = os.path.join(logs_dir, "app.log")

# Initialize the rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=100 * 1024 * 1024,  # 100 MB
    backupCount=5,
)

# Configure file handler log format and level
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Create a console handler with a specific log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Get the root logger and attach handlers directly
root_logger = logging.getLogger()

# Add both the file handler and console handler to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set the overall logging level (root level)
root_logger.setLevel(logging.DEBUG)
