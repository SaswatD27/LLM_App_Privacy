from ..import_utils import *
from ..custom_utils import *

class attackLoader:
    def __init__(self):
        pass

    def get_attack_prompt(self, attack_type='none', prompt_index=1):
        attack_prompts = load_yaml('/bigtemp/duh6ae/LLM_App_Privacy/src/utils/attack/attack_dictionary.yaml')
        try:
            return attack_prompts[attack_type][f'type{prompt_index}']['prompt']
        except:
            return attack_prompts[attack_type][f'{prompt_index}']['prompt']