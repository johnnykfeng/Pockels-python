import sys
from loguru import logger

logger.remove()  # Remove default logger

# sys.stdout is the console output, level="TRACE" means log everything
logger.add(sink=sys.stdout, level="TRACE")
# setting for logfile
logger.add("loguru_pack/logfile.log", level="WARNING", rotation="12:00") # Create new file at 12AM
