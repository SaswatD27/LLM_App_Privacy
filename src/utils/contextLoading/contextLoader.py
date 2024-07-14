from ..import_utils import *
from ..custom_utils import *

local_data_path = '/scratch/duh6ae/LLM_App_Privacy/local_data'

class contextLoader:
    def __init__(self, data = f'{local_data_path}/adult', data_attribute = "text"):
        self.data = data
        self.loaded_data = self.load_data()
        self.data_attribute = data_attribute

    def load_data(self, split = 'test'):
        try:
            dataset = load_from_disk(self.data)[split]
        except:
            dataset = load_dataset(self.data)[split]
        return dataset
    
    def load_context(self, data_index):
        return self.loaded_data[data_index][self.data_attribute]
