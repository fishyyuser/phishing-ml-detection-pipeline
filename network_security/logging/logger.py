import logging
import os
from datetime import datetime

# Create global logs directory
LOGS_BASE_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_BASE_DIR, exist_ok=True)

# Create timestamped log folder only once per run
RUN_TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
LOG_DIR = os.path.join(LOGS_BASE_DIR, RUN_TIMESTAMP)
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path
LOG_FILE = f"{RUN_TIMESTAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging (only once)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create logger instance for the package
logger = logging.getLogger("network_security")
logger.setLevel(logging.INFO)
