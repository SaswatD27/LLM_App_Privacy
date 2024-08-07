from .import_utils import *
from .defense.defenseLoader import defenseLoader
from .attack.attackLoader import attackLoader
from .contextLoading.contextLoader import contextLoader

class agent:
    def __init__(self, model_name, context = None, cache_dir = '/bigtemp/duh6ae/hfhub_cache', max_new_tokens = 250, do_sample = False, repetition_penalty = 1.03, safety_prompt_index = 1, attack_prompt_type = 'none', attack_prompt_index = 1, context_data = '/bigtemp/duh6ae/LLM_App_Privacy/local_data/adult', context_data_attribute = 'text', context_data_index = 0, predefenses=["query_rewriter"], postdefenses=["check_jailbroken"], sensitive_attributes = ["race", "age", "education", "marital status", "gender"], attributes = [], num_shots = 0, fetch_probs = False):
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
        self.num_shots = num_shots

        if context is not None:
            self.context = context
        else:
            self.context = self.contextLoader.load_context(context_data_index, self.num_shots)

        self.attributes = self.get_attributes()
        
        self.template = "<|user|>\n{context}\n{safety_prompt}\nAnswer this question: {input_text}. \n<|assistant|>\nAnswer:"
        
        self.predefenses = self.defenseLoader.predefenses
        self.postdefenses = self.defenseLoader.postdefenses
        self.attack_prompt_type = attack_prompt_type
        self.attack_prompt_index = attack_prompt_index

        self.fetch_probs = fetch_probs

    def get_attributes(self): # works for the adult data, might need to tweak if context format changes
       context_shards = self.context.split('.')
       attribute_list = []
       for shard in context_shards:
            if ' is ' in shard:
                attribute_list.append(shard.split(' is ')[0].lower()[1:])
       print(f'attribute_list: {attribute_list}')
       assert len(attribute_list) > 0, "No attributes found in context"
       # print(attribute_list)
       return attribute_list        
    
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
    
    def get_probs(self, outputs, token_list, input_len):
        # Extract the logits from the output
        logits = outputs.scores
        cumulative_prob = 0
        for step, step_logits in enumerate(logits):
            # Convert logits to probabilities
            probabilities = torch.softmax(step_logits[0], dim=-1)
            
            # Get all token IDs and their corresponding probabilities
            token_ids = torch.arange(probabilities.size(-1)).tolist()
            prob_values = probabilities.tolist()
            
            # Decode the token IDs to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            target_token_id_list = [self.tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in token_list]
            # Print the step number, tokens, and their corresponding probabilities
            print(f"\nStep {step + 1}:")
            for token, token_id, prob in zip(tokens, token_ids, prob_values):
                if token in token_list or token_id in target_token_id_list:
                    print(f"Token: {token}, Probability: {prob}")
                    if step + 1 == 1:
                        cumulative_prob += prob
        return cumulative_prob

    
    def generate_response(self, input_text):
        prompt = self.template.format(context=self.context, safety_prompt=self.safety_prompt, input_text=input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.device)
        input_len = inputs.input_ids.shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, return_dict_in_generate=True, output_scores=True)
        #print(outputs)
        ###
        print(f'Fetch Label Logits?: {self.fetch_probs}')
        if self.fetch_probs:
            cumulative_prob_dict = {'options': [], 'sensitive': [], 'non_sensitive': []}
            option_token_list = ['yes', 'no']
            sensitive_token_list = []
            non_sensitive_token_list = []
            for att in self.attributes:
                # print(f'Sensitive Attributes: {self.sensitive_attributes}')
                # print(f'All Attributes: {self.attributes}')
                # print(f'Attribute: {att}')
                # print(f'Attribute in sensitive?: {att in self.sensitive_attributes}')
                # print(self.context)
                if att in self.sensitive_attributes:
                    try:
                        sensitive_token_list.append(self.context.lower().split(f'{att} is ')[1].split('.')[0])
                    except:
                        print(f'Error with attribute: {att}. Perhaps absent.')
                        #print(self.context.lower().split(f'{att} is '))
                        pass
                else:
                    try:
                        non_sensitive_token_list.append(self.context.lower().split(f'{att} is ')[1].split('.')[0])
                    except:
                        print(f'Error with attribute: {att}. Perhaps absent.')
                        #print(self.context.lower().split(f'{att} is '))
                        pass
            print('##################\n OPTION TOKENS')
            cumulative_prob_dict['options'] = [self.get_probs(outputs, option_token_list, input_len)]
            print('SENSITIVE TOKENS')
            print(sensitive_token_list)
            cumulative_prob_dict['sensitive'] = [self.get_probs(outputs, sensitive_token_list, input_len)]
            print('NON-SENSITIVE TOKENS')
            cumulative_prob_dict['non_sensitive'] = [self.get_probs(outputs, non_sensitive_token_list, input_len)]
            print('##################')
            print(f'Cumulative Probabilities: {cumulative_prob_dict}')
        #output_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f'Output Text: {output_text}')
        #delete inputs and outputs from the gpu memory

        del inputs
        del outputs
        if self.fetch_probs:
            return output_text, cumulative_prob_dict
        else:
            return output_text

    def run(self, input_text):
        # Adversary applies the attack prompt to the query
        input_text = self.apply_attack_prompt(input_text)
        # Query passes through prefilters
        self.predefense(input_text)
        # Query is sent to the model and response is generated
        if self.fetch_probs:
            response, cumulative_prob_dict = self.generate_response(input_text)
        else:
            response = self.generate_response(input_text)
        # print(f'Unfiltered Response: {response} OVER')
        try:
            # try:
            response = response.split("Answer:")[1].strip()
            # except:
            #     print(response)
            #     exit()
        except IndexError:
            pass
        # Response passes through postfilters and output
        returned_response = self.postdefense(response)
        if self.fetch_probs:
            return returned_response, cumulative_prob_dict
        else:
            return returned_response