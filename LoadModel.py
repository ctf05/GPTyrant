from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os


def load_model(model):
    model_directory = os.path.join("Models", model)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    # Set tokenizer to pad on the left
    tokenizer.padding_side = "left"
    # Set the pad token to EOS token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_directory)
    return model, tokenizer