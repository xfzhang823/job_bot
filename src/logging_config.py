# logging_config.py

import logging
import os
from utils.generic_utils import find_project_root


# Ensure logs directory exists

# Determine the root directory by moving up from the current script location
root_dir = find_project_root()

# Determine the path to the logs directory in the root directory
logs_dir = os.path.join(root_dir, "logs")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Configure the root logger
logging.basicConfig(
    filename=os.path.join(logs_dir, "app.log"),  # Use full path to logs directory
    level=logging.DEBUG,  # Log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
)

# Create a console handler with a specific log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for the console handler
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)
