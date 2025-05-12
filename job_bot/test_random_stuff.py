import logging
import logging_config
from datetime import datetime

logger = logging.getLogger(__name__)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(f"teting logger at {current_time}")
