from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

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
print("Logit Scores:", logits)
print("Length of Logit Scores:", [len(logit[0]) for logit in logits])
