import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_gpt2(model_name="gpt2", save_directory="Models/gpt2"):

    if (os.path.exists(save_directory)):
        return

    # Ensure the save directory exists
    os.makedirs(save_directory)

    # Load pretrained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Save the model and the tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

# Example usage
download_gpt2("gpt2", "Models/gpt2")
