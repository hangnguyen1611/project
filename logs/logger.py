import logging
from pathlib import Path

LOG_FILE = "../logs/train_model.log"

def setup_logger(level=logging.INFO):
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Xóa tất cả StreamHandler
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]


    # Thêm FileHandler
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(fh)

    # Không thêm console handler
    root_logger.propagate = False

    return root_logger
