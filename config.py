import os
from datetime import datetime

EPOCH = 300

DATA_DIR = 'F:\Project\deep_learning\data'

CHECKPOINT_PATH = 'checkpoint'

SAVE_EPOCH = 10

MILESTONES = [60, 120, 160]

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'