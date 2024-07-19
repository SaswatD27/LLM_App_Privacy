from utils.import_utils import *
from utils.custom_utils import *
from utils.agentUtils import agent

if __name__ == '__main__':
    model_name = "allenai/tulu-2-dpo-7b" #"allenai/tulu-v2.5-dpo-13b-uf-mean"
    app_agent = agent(model_name=model_name, max_new_tokens = 5)
    input_text = "What is the race of this person?"
    response = app_agent.run(input_text)
    print(f'Response: {response}')