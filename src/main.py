
from trainer import ModelTrainer
import logging
import os
import psutil
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Consumer")
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--output_file', type=str, required=False, default=None)
    parser.add_argument('--model_file', type=str, required=False, default=None)
    args = parser.parse_args()
    log_dir = args.log_dir
    output_file = args.output_file
    model_file = args.model_file

    logger = logging.getLogger(__name__)
    logging.root.setLevel(level=logging.INFO)
    model_trainer = ModelTrainer(log_dir=log_dir, model_file=model_file, output_file=output_file)
    logger.info('model trainer initialized.')
    model_trainer.model_train()
    logger.info('model trainer done.')
    print("Memory: ", psutil.Process(os.getpid()).memory_info().rss)
