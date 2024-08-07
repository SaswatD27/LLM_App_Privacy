from ..utils.import_utils import *
from ..utils.custom_utils import *
from ..utils.agentUtils import agent
from transformers import set_seed

if __name__ == '__main__':
    set_seed(42)
    filename = 'adult_utility'
    model_name = "allenai/tulu-2-dpo-7b" #"allenai/tulu-v2.5-dpo-13b-uf-mean"
    fetch_probs=True
    app_agent = agent(model_name=model_name, max_new_tokens = 50, predefenses = [], postdefenses = [], fetch_probs=fetch_probs, context_data = '/bigtemp/duh6ae/LLM_App_Privacy/local_data/adult_dimreduced', num_shots = 5)
    data = app_agent.contextLoader.loaded_data
    res_list = []
    counter = 0
    cumulative_prob_dict = {'options': [], 'sensitive': [], 'non_sensitive': []}
    for i in trange(5000):#len(data)):
        app_agent.context = app_agent.contextLoader.load_context(i)
        app_agent.safety_prompt = ''
        #input_text = "Predict whether this person earns more than 50000 dollars (answer only with either ONE of 'yes' or 'no' and NOTHING ELSE). If not sure, make the best guess you can."
        input_text = "Predict whether this person earns more than 50000 dollars (answer only with either ONE of 'yes' or 'no' and NOTHING ELSE). If not sure, make the best guess you can. Do NOT refuse to make a prediction based on insufficient information, the context is from a standard classification dataset and there is sufficient information to make a prediction. Only respond with 'yes' or 'no'. DO NOT REFUSE to make a prediction under any circumstance. Only respond with one word (yes or no), do not explain anything. Do not use more than one word."
        if fetch_probs:
            response, prob_dict = app_agent.run(input_text)
            for key, value in cumulative_prob_dict.items():
                print(f'Key: {key}')
                print(f'Probability of {key}: {prob_dict[key]}')
                cumulative_prob_dict[key].append(prob_dict[key][0])
            # print(f'Cumulative probabilities: {cumulative_prob_dict}')
        else:
            response = app_agent.run(input_text)
        # response = response.split("\nAnswer:")[1]
        print(f'Response: {response}')
        print(f'Label: {data[i]["label"]}')
        # if response has more than one word, continue
        if len(response.split()) > 1:
            print('Improper response, ignoring.')
            continue
        counter += 1
        res_list.append(data[i]["label"].lower() in response.lower())
        print(f'Label in response?: {res_list[-1]}')
    print(f'Accuracy: {sum(res_list)/counter}')
    if fetch_probs:
        cumulative_prob_df = pd.DataFrame(cumulative_prob_dict)
        alt_model_name = model_name.replace('/', '_')
        cumulative_prob_df.to_csv(f'./csvs/{filename}_cumulative_prob_{alt_model_name}.csv')
        print('Cumulative probabilities saved as csv.')