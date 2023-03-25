import logging
from midjourney_model import construct_index

folder_name = 'midjourney'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info(f"Constructing index for folder {folder_name}...")
index = construct_index(folder_name)
if index is None:
    logging.error("Failed to construct index.")
else:
    logging.info("Index constructed successfully.")
