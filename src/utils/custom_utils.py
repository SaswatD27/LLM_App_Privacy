from .import_utils import *

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_text(input_text, model, tokenizer, max_new_tokens = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens = max_new_tokens)#, num_return_sequences=k)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text