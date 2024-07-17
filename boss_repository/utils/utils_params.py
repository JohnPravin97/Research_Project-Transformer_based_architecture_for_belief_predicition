import os
import datetime
import logging

def gen_run_folder(path_model_id='./logging_run'):
    run_paths = dict()
    date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
    run_id = 'run_' + date_creation
    run_paths['path_model_id'] = os.path.join(path_model_id, run_id)
    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'run.log')

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths

def set_loggers(path_log=None, logging_level=0):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)

if __name__=='__main__':
    assert("This can not be run as a single file")