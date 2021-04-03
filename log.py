import logging
import sys
# from logging.handlers import FileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = "my_app.log"

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    console_handler.setLevel(logging.INFO)
    return console_handler

def get_file_handler(filename):
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name,filename, log_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)

    logger.setLevel(log_level) # better to have too much log than not enough

    # logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(filename))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger
