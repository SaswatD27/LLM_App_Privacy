from .import_utils import *
from .defense.defenseLoader import defenseLoader
from .attack.attackLoader import attackLoader
from .contextLoading.contextLoader import contextLoader

class agent:
    def __init__(self, model_name, context = None, cache_dir = '/bigtemp/duh6ae/hfhub_cache', max_new_tokens = 250, do_sample = False, repetition_penalty = 1.03, safety_prompt_index = 1, attack_prompt_type = 'none', attack_prompt_index = 1, context_data = '/bigtemp/duh6ae/LLM_App_Privacy/local_data/adult', context_data_attribute = 'text', context_data_index = 0, predefenses=["query_rewriter"], postdefenses=["check_jailbroken"], sensitive_attributes = ["race", "age", "education", "marital status"], fetch_logits = False):
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
        
        self.template = "<|user|>\n{context}\n{safety_prompt}\nAnswer this question: {input_text}. \n<|assistant|>\nAnswer:"
        
        self.predefenses = self.defenseLoader.predefenses
        self.postdefenses = self.defenseLoader.postdefenses
        self.attack_prompt_type = attack_prompt_type
        self.attack_prompt_index = attack_prompt_index

        self.fetch_logits = fetch_logits
    
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
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, return_dict_in_generate=True, output_scores=True)
        #print(outputs)
        ###
        print(f'Fetch Label Logits?: {self.fetch_logits}')
        if self.fetch_logits:
            print('Fetching Label Logits!')
            # Token IDs for "yes" and "no" (case insensitive)
            yes_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in ['yes']] #['yes', 'Yes', 'YES']
            no_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in ['no']] #['no', 'No', 'NO']

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            print(f"generated_tokens:{generated_tokens}")
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                # | token | token string | logits | probability
                decoded_tok = self.tokenizer.decode(tok).lower()
                if 'yes' in decoded_tok or 'no' in decoded_tok:
                    print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
                        
            # # Extract logits for "yes" and "no" from outputs.scores
            # yes_logits = []
            # no_logits = []
            
            # for scores in outputs.scores:
            #     # Find logits for "yes" tokens
            #     yes_logits.append(scores[-1, yes_token_ids].detach().cpu().numpy())
            #     # Find logits for "no" tokens
            #     no_logits.append(scores[-1, no_token_ids].detach().cpu().numpy())
            
            # # Average logits across all positions
            # yes_logits = np.mean(np.concatenate(yes_logits, axis=0), axis=0)
            # no_logits = np.mean(np.concatenate(no_logits, axis=0), axis=0)
            # # logits = outputs.scores[0] #outputs.logits if hasattr(outputs, 'logits') else
            # # print(len(logits), len(outputs.sequences[0]))
            # # yes_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in ['yes', 'Yes', 'YES']]
            # # no_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in ['no', 'No', 'NO']]
            
            # # yes_logits = logits[:, yes_token_ids].mean(dim=-1).detach().cpu().numpy()
            # # no_logits = logits[:, no_token_ids].mean(dim=-1).detach().cpu().numpy()
            
            # print(f"Logits for 'yes': {yes_logits}")
            # print(f"Logits for 'no': {no_logits}")
        ###
        #output_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        #delete inputs and outputs from the gpu memory
        del inputs
        del outputs
        return output_text

    def run(self, input_text):
        # Adversary applies the attack prompt to the query
        input_text = self.apply_attack_prompt(input_text)
        # Query passes through prefilters
        self.predefense(input_text)
        # Query is sent to the model and response is generated
        response = self.generate_response(input_text)
        # print(f'Unfiltered Response: {response} OVER')
        try:
            # try:
            response = response.split("answer:")[1].strip()
            # except:
            #     print(response)
            #     exit()
        except IndexError:
            pass
        # Response passes through postfilters and output
        return self.postdefense(response)