from concurrent import futures
import os
import traceback
from datetime import datetime
import time
import logging
import pytz


def exception_on_one_line(exception):
    summary = traceback.extract_tb(exception.__traceback__)
    frame = summary[-1]
    filename = frame.filename
    lineno = frame.lineno
    return f"{filename}:{lineno}: {type(exception).__name__}: {exception}\n"

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

class PrefixedFormatter(logging.Formatter):

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        dt = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Seoul'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s

    def format(self, record):
        record.prefix = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_message = super().format(record)
        if record.levelno == logging.ERROR:
            return f"[{record.prefix}] {RED}{log_message}{RESET}"
        elif record.levelno == logging.WARNING:
            return f"[{record.prefix}] {YELLOW}{log_message}{RESET}"
        elif record.levelno == logging.DEBUG:
            return f"[{record.prefix}] {GREEN}{log_message}{RESET}"
        else:
            return f"[{record.prefix}] {RESET}{log_message}{RESET}"

def createFolder(directory):
    os.makedirs(directory, exist_ok=True)

def get_timestamp():
    now = datetime.now()
    formattedDate = now.strftime("%Y%m%d_%H%M%S")
    return str(formattedDate)

def get_timestamp_day():
    now = datetime.now()
    formattedDate = now.strftime("%Y%m%d")
    return str(formattedDate)

def make_logger(loger_path):

    createFolder(loger_path)
    logger = logging.getLogger("my")
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = PrefixedFormatter('%(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
    stream_hander = logging.StreamHandler()	
    stream_hander.setFormatter(formatter)		
    logger.addHandler(stream_hander)		
    file_handler = logging.FileHandler(loger_path + '/' + get_timestamp_day() +'.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger		

def get_all_image(root_dir):
    possible_img_extension = ['.jpg', '.png', '.bmp', '.PNG', '.Png', '.gif', '.jpge']
    list=[]
    for (root, dirs, files) in os.walk(root_dir):
        if len(files)>0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    _, ext = os.path.splitext(file_name)
                    list.append(file_name)
    return list