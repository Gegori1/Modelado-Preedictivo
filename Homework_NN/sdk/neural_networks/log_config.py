import logging

def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    f_handler = logging.FileHandler('logging_outputs.txt')
    i_handler = logging.StreamHandler()  # Add info handler
    # c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)
    i_handler.setLevel(logging.INFO)  # Set info handler level to INFO

    f_format = logging.Formatter('%(pathname)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    i_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')  # Format for info handler

    f_handler.setFormatter(f_format)
    i_handler.setFormatter(i_format)  # Set formatter for info handler

    logger.addHandler(f_handler)
    logger.addHandler(i_handler)  # Add info handler to logger

    return logger

import functools

def with_logging(info_message=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = configure_logger(func.__module__)
            try:
                result = func(*args, **kwargs)
                if info_message is not None:
                    logger.info(info_message)  # Use the info_message parameter as the info message
                return result
            except Exception as e:
                # Log the exception
                logger.error(f"Exception occurred in function '{func.__name__}' of module '{func.__module__}': {e}")
                # Re-raise the exception
        return wrapper
    return decorator
