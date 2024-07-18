from ..utils.import_utils import *
from ..utils.custom_utils import *
from ..utils.agentUtils import agent

if __name__ == '__main__':
    model_name = "allenai/tulu-2-dpo-7b" #"allenai/tulu-v2.5-dpo-13b-uf-mean"
    app_agent = agent(model_name=model_name, max_new_tokens = 10)
    data = app_agent.contextLoader.loaded_data
    for i in range(len(data)):
        app_agent.context = app_agent.contextLoader.load_context(i)
        input_text = "Does this person earn more than 50000 dollars (yes or no)?"
        response = app_agent.generate_response(input_text)
        print(f'Response: {response}')