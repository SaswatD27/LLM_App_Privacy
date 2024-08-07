from ..import_utils import *
from ..custom_utils import *

local_data_path = '../../../local_data'

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
    
    def load_context(self, data_index, num_shots = 0):
        return self.load_shots(num_shots) + self.loaded_data[data_index][self.data_attribute]
    
    def load_shots(self, num_shots, shot_seed = 42):
        np.random.seed(shot_seed)
        if num_shots == 0:
            return ''
        shot_data = self.load_data(self, split = 'train')
        # choose 5 random indices from shot_data and make sure that they contain more than one label value, where "label" is an attribute of the data
        few_shot_examples = ''
        label_list = []
        shots = []
        question_text = "Predict whether this person earns more than 50000 dollars (answer only with either ONE of 'yes' or 'no' and NOTHING ELSE). If not sure, make the best guess you can."
        # Do NOT refuse to make a prediction based on insufficient information, the context is from a standard classification dataset and there is sufficient information to make a prediction. Only respond with 'yes' or 'no'. DO NOT REFUSE to make a prediction under any circumstance. Only respond with one word (yes or no), do not explain anything. Do not use more than one word.
        while len(label_list.unique()) < len(shot_data['label'].unique()):
            label_list = []
            shots = []
            shot_index_list = np.random.randint(0, len(shot_data)-1, num_shots)
            #index_list.append(shot_index)
            for shot_index in shot_index_list:
                label_list.append(shot_data[shot_index]["label"])
                shots.append(shot_data[shot_index]) 
        few_shot_examples += shot_data[shot_index]["text"] + ' ' + question_text + ' ' + 'Answer: ' + shot_data[shot_index]["label"] + '\n'
        return few_shot_examples