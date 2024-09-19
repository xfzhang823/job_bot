import logging
import logging.handlers
import os
from utils.generic_utils import find_project_root

# Ensure logs directory exists
root_dir = find_project_root()
logs_dir = os.path.join(root_dir, "logs")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Set up log file rotation: max 10MB per file, up to 5 backup files
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(logs_dir, "app.log"),
    maxBytes=100 * 1024 * 1024,
    backupCount=5,  # 100 MB
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

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Log level
    handlers=[file_handler, console_handler],  # Use both file and console handlers
)
