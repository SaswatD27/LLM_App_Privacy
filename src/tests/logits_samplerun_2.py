from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Encode the input text
input_text = "Translate English to French: How are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate with output_scores and return_dict_in_generate set to True
output = model.generate(
    input_ids,
    max_length=20,
    output_scores=True,
    return_dict_in_generate=True
)

# Extract the logits from the output
logits = output.scores

# Decode the generated tokens
generated_tokens = output.sequences
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

print("Generated Text:", decoded_text)

# Extract and display all candidate tokens and their probabilities at each step
for step, step_logits in enumerate(logits):
    # Convert logits to probabilities
    probabilities = torch.softmax(step_logits[0], dim=-1)
    
    # Get all token IDs and their corresponding probabilities
    token_ids = torch.arange(probabilities.size(-1)).tolist()
    prob_values = probabilities.tolist()
    
    # Decode the token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Print the step number, tokens, and their corresponding probabilities
    print(f"\nStep {step + 1}:")
    for token, prob in zip(tokens, prob_values):
        #print(prob, type(prob), prob > 0.0)
        if prob > 0.0001:
            print(f"Token: {token}, Probability: {prob}")
