from .import_utils import *
from .defense.defenseLoader import defenseLoader
from .attack.attackLoader import attackLoader
from .contextLoading.contextLoader import contextLoader

class agent:
    def __init__(self, model_name, context = None, cache_dir = '/bigtemp/duh6ae/hfhub_cache', max_new_tokens = 250, do_sample = False, repetition_penalty = 1.03, safety_prompt_index = 1, attack_prompt_type = 'none', attack_prompt_index = 1, context_data = '/bigtemp/duh6ae/LLM_App_Privacy/local_data/adult', context_data_attribute = 'text', context_data_index = 0, predefenses=["query_rewriter"], postdefenses=["check_jailbroken"], sensitive_attributes = ["race", "age", "education", "marital status"]):
        os.environ['HF_HOME'] = cache_dir
        self.predefenses = predefenses
        self.postdefenses = postdefenses
        self.context_data = context_data
        self.context_data_attribute = context_data_attribute # Which attribute of the loaded data to use as context; for adult, it's "text"

        self.defenseLoader = defenseLoader(predefenses=self.predefenses, postdefenses=self.postdefenses)
        self.attackLoader = attackLoader()
        self.contextLoader = contextLoader(self.context_data, self.context_data_attribute)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.repetition_penalty = repetition_penalty
        self.safety_prompt = self.defenseLoader.get_safety_prompt(safety_prompt_index)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sensitive_attributes = sensitive_attributes

        if context is not None:
            self.context = context
        else:
            self.context = self.contextLoader.load_context(context_data_index)
        
        self.template = "<|user|>\n{context}\n{safety_prompt}\nAnswer this question: {input_text}. \n<|assistant|>\n"
        
        self.predefenses = self.defenseLoader.predefenses
        self.postdefenses = self.defenseLoader.postdefenses
        self.attack_prompt_type = attack_prompt_type
        self.attack_prompt_index = attack_prompt_index
    
    def get_predefenses(self):
        return self.defenseLoader.predefenses
    
    def get_postdefenses(self):
        return self.defenseLoader.postdefenses

    def predefense(self, input_text):
        defenses = self.get_predefenses()
        for defense in defenses:
            input_text = defense(self, input_text)
        return input_text

    def postdefense(self, response):
        defenses = self.get_postdefenses()
        for defense in defenses:
            response = defense(self, response)
        return response
    
    def apply_attack_prompt(self, input_text): #CHECK
        attack = self.attackLoader.get_attack_prompt(self.attack_prompt_type, self.attack_prompt_index)
        input_text = attack + '\n' + input_text
        return input_text
    
    def generate_response(self, input_text):
        prompt = self.template.format(context=self.context, safety_prompt=self.safety_prompt, input_text=input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return output_text

    def run(self, input_text):
        # Adversary applies the attack prompt to the query
        input_text = self.apply_attack_prompt(input_text)
        # Query passes through prefilters
        self.predefense(input_text)
        # Query is sent to the model and response is generated
        response = self.generate_response(self, input_text)
        try:
            response = response.split("answer:")[1].strip()
        except IndexError:
            pass
        # Response passes through postfilters and output
        return self.postdefense(response)