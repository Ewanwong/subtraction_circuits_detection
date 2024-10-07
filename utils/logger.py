""" Logger utility.
"""
import getpass
import logging
import os
import time


def get_logger(curr_time, user, save=True):
    dir_path = 'logs/{}/'.format(user)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s: %(lineno)d: %(levelname)s: %(message)s',
                        filename=dir_path+'{}.log'.format(curr_time) if save else None,
                        filemode='w' if save else None)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(filename)s: %(lineno)d: %(levelname)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # add the handler to the root logger
        logger.addHandler(console)
    # logger.propagate = False
    return logger


username = getpass.getuser()
log = get_logger(time.time(), username)