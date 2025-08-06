import requests
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Enable logging
logging.basicConfig(level=logging.INFO)

# Define your IP location tool
def get_ip_location():
    url = "https://ipinfo.io/json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return (
                f"Your IP: {data['ip']}\n"
                f"City: {data.get('city', 'N/A')}\n"
                f"Country: {data.get('country', 'N/A')}\n"
                f"Coordinates: {data.get('loc', 'N/A')}"
            )
        else:
            return f"Failed with status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Load the Flan-T5 model locally
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create pipeline for text generation
llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Function to ask LLM if input is about IP
def detect_ip_intent(user_input):
    intent_prompt = f"Is the following request asking about the user's IP address or location? Answer with only Yes or No.\n\nRequest: {user_input}"
    response = llm_pipe(intent_prompt, max_new_tokens=5, do_sample=False, temperature=0.0)
    answer = response[0]["generated_text"].strip().lower()
    logging.info(f"Intent detection: {answer}")
    return "yes" in answer

# Main logic
def handle_user_input(user_input):
    if detect_ip_intent(user_input):
        return get_ip_location()
    else:
        # General response from the LLM
        result = llm_pipe(user_input, max_new_tokens=150, do_sample=False, temperature=0.7)
        return result[0]["generated_text"]

# Example usage
if __name__ == "__main__":
    prompt = input("You: ")
    response = handle_user_input(prompt)
    print(f"\n🤖 Response:\n{response}")
