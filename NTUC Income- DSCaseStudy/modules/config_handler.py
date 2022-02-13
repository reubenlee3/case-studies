import os
import json
from datetime import datetime

class ConfigHandling:
    def __init__(self, job_directory, config_filename='config.json'):
        config = []
        with open(os.path.join(job_directory, config_filename)) as config_file:
            config = json.load(config_file)

        self.current_directory = job_directory
        self.train_data_path = os.path.join(self.current_directory, config["data_directory"], config["train_data"])
        self.test_data_path = os.path.join(self.current_directory, config["data_directory"], config["test_data"])
        self.train_data_clean_path = os.path.join(self.current_directory, config["data_directory"], 'train_data_clean.csv')
        self.test_data_clean_path = os.path.join(self.current_directory, config["data_directory"], 'test_data_clean.csv')
