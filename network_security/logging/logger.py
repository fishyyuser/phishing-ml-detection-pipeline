import logging
import os
from datetime import datetime

# Create timestamped log folder
LOG_FOLDER = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
LOG_DIR = os.path.join(os.getcwd(), "logs", LOG_FOLDER)
os.makedirs(LOG_DIR, exist_ok=True)

# Define full log file path
LOG_FILE = f"{LOG_FOLDER}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)