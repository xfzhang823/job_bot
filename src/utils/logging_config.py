"""logging_config.py"""

import logging
import os

# Creat logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


# Configure logging
logging.basicConfig(
    filename=os.path.join("logs", "app.log"),
    level=logging.debug,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format)
)


# Create a console handler for logging to the console as well
# (a handler is part of the logging infrastructure and is responsible for determining
# where the logs should be output, such as to a file, console, email, remote server, etc.)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set level to DEBUG for consol output

# Define a formatter for console output
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)
