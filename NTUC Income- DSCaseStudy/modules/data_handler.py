import os
import pandas as pd
import datetime

class DataHandling:

    def __init__(self, cfg):
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data_clean_path = cfg.train_data_clean_path
        self.test_data_clean_path = cfg.test_data_clean_path


    def read_data(self, dataset='train'):
        if dataset == 'train':
            data = pd.read_csv(self.train_data_path)
            data = data.astype({
                'cust_id': 'int32',
                'gender': 'O',
                'age': 'int32',
                'driving_license': 'int32',
                'region_code': 'int32',
                'previously_insured': 'int32',
                'vehicle_age': 'O',
                'vehicle_damage': 'O',
                'annual_premium': 'float32',
                'policy_sales_channel': 'int32',
                'days_since_insured': 'int32',
                'response': 'int32'
            })
        elif dataset == 'train_clean':
            data = pd.read_csv(self.train_data_clean_path)
        elif dataset == 'test_clean':
            data = pd.read_csv(self.test_data_clean_path)
        else:
            data = pd.read_csv(self.test_data_path)
            data = data.astype({
                'cust_id': 'int32',
                'gender': 'O',
                'age': 'int32',
                'driving_license': 'int32',
                'region_code': 'int32',
                'previously_insured': 'int32',
                'vehicle_age': 'O',
                'vehicle_damage': 'O',
                'annual_premium': 'float32',
                'policy_sales_channel': 'int32',
                'days_since_insured': 'int32'
            })
        
        print(f"{dataset} has shape: {data.shape}")
        return data

