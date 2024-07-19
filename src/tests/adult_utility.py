from ..utils.import_utils import *
from ..utils.custom_utils import *
from ..utils.agentUtils import agent

if __name__ == '__main__':
    model_name = "allenai/tulu-2-dpo-7b" #"allenai/tulu-v2.5-dpo-13b-uf-mean"
    app_agent = agent(model_name=model_name, max_new_tokens = 10, predefenses = [], postdefenses = [])
    data = app_agent.contextLoader.loaded_data
    res_list = []
    for i in trange(len(data)):
        app_agent.context = app_agent.contextLoader.load_context(i)
        app_agent.safety_prompt = ''
        input_text = "Does this person earn more than 50000 dollars (yes or no)?"
        response = app_agent.run(input_text)
        print(f'Response: {response}')
        print(f'Label: {data[i]["label"]}')
        res_list.append(data[i]["label"].lower() in response.lower())
        print(f'Label in response?: {res_list[-1]}')
    print(f'Accuracy: {sum(res_list)/len(res_list)}')