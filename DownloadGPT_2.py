import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_gpt2(model_name="gpt2", save_directory="Models/gpt2"):

    if (os.path.exists(save_directory)):
        return

    os.makedirs(save_directory)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
