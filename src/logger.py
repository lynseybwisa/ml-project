import logging #module for emitting log messages
import os 
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # current date and time formatted as 'month_day_year_hour_minute_second.log'
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #path where the log file will be saved. 
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) #path to the log file (including the file name)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, #  only log messages with an INFO level or higher (e.g., INFO, WARNING, ERROR, CRITICAL) 
)

if __name__=="__main__":
    logging.info("Logging has started")
