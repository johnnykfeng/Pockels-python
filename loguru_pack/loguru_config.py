import sys
from loguru import logger

# logger.remove(0)
# logger.add(sink=sys.stderr, format="{time} {level} {message}", filter="my_module", level="WARNING")

logger.add("loguru_pack/logfile.log", rotation="12:00") # Create new file at 12AM
# logger.add("loguru_pack/logfile.log", rotation="1 MB") # Automatically rotate too big file
# logger.add("loguru_pack/logfile.log", rotation="1 week") # Cleanup after a week

logger.trace("*** Initiating loguru ***")

def try_all_levels(set_level="SUCCESS"):

    logger.add(sink = sys.stderr,
            format="{time} {level} {message}",
            # filter="my_module", 
            level=set_level)
    
    logger.add(sink = sys.stdout,
        format="{time} {level} {message}",
        # filter="my_module", 
        level=set_level)
    
    logger.add("loguru_pack/logfile.log", 
               level=set_level)

    logger.trace("this is TRACE")
    logger.debug("this is DEBUG")
    logger.info("this is INFO")
    logger.success("this is SUCCESS")
    logger.warning("this is WARNING")
    logger.error("this is ERROR")
    logger.critical("this is CRITICAL")

# try_all_levels(set_level="TRACE")

def bind_example():
    logger.add("loguru_pack/logfile.log", format="{extra[ip]} {extra[user]} {message}")
    context_logger = logger.bind(ip="192.168.0.1", user="someone")
    context_logger.info("Contextualize your logger easily")
    context_logger.bind(user="someone_else").info("Inline binding of extra attribute")
    context_logger.info("Use kwargs to add context during formatting: {user}", user="anybody")


# logger.info("If you're using Python {}, prefer {feature} of course!", 3.6, feature="f-strings")

# logger.level("FATAL", no=60, color="<red>", icon="!!!")
# logger.log("FATAL", "A user updated some information.")
