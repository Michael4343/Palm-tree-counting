import shutil
import os
import logging

logging.basicConfig(level=logging.INFO)

def safe_delete(path):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            logging.info(f"Directory {path} successfully deleted.")
        except Exception as e:
            logging.error(f"Failed to delete directory {path}: {e}", exc_info=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
            logging.info(f"File {path} successfully deleted.")
        except Exception as e:
            logging.error(f"Failed to delete file {path}: {e}", exc_info=True)
    else:
        logging.warning(f"Path {path} does not exist. No action taken.")