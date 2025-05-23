from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a small, free model that works on your computer
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the tool to break text into pieces (tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model to answer questions
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use less memory
    device_map="auto",  # Put model on GPU if possible
    load_in_4bit=True  # Shrink model to fit your GPU
)

# Ask a question
prompt = "What is a ROC curve?"

# Turn the question into numbers for the model
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Get the answer
outputs = model.generate(**inputs, max_length=100)

# Turn the answer back into text
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Show the answer
print(f"Question: {prompt}")
print(f"Answer: {answer}")