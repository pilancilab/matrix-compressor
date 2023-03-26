from hlb.data import create_dataset
from loguru import logger
from hlb.config import hyp

if __name__ == "__main__":
    logger.info(f"Initiating dataset creation")
    data = create_dataset()
    logger.info(f"Finished writing dataset to {hyp['misc']['data_location']}")
