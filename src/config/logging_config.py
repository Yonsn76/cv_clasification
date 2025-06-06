import logging

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='[%(levelname)s] %(name)s: %(message)s'
)
