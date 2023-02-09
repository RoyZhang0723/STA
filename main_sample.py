import time
import os
import logging

from config import *
config = Config()

os.environ["CUDA_VISIBLE_DEVICES"] = config.use_gpu

from dataset import *
from train import *
from evaluate import *
from tool_logger import *


def main_sample():
    LOG_FP = '../log/' +  \
        config.dataset_script[11:-3] + '_' + \
        config.model_name.split('/')[-1] + '_' + \
        time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + \
        '.txt'
    logger.setlogger(None, LOG_FP)

    for name, value in vars(config).items():
        logging.info('$$$$$ custom para {}: {}'.format(name, value))

    dataset = load_dataset_for_transformer(config)
    config.train_dataloader = dataset.train_dataloader
    config.eval_dataloader = dataset.test_dataloader

    training(config)

main_sample()
