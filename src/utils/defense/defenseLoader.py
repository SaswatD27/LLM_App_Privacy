from .defenseUtils import *

class defenseLoader:
    def __init__(self, predefenses=["query_rewriter"], postdefenses=["check_jailbroken"]):
        self.predefenses = [defense_dict['pre'][defense] for defense in predefenses]
        self.postdefenses = [defense_dict['post'][defense] for defense in postdefenses]
    
    def get_safety_prompt(self, prompt_index=1):
        return safety_prompts[prompt_index]