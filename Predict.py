import torch
import nltk
from nltk import pos_tag
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def predict(input_text, model, tokenizer, method):
    if method == "Best":
        return predict_next_word(input_text, model, tokenizer)
    elif method == "Worst":
        return predict_least_likely_word(input_text, model, tokenizer)
    elif method == "Coherent Worst":
        return predict_coherent_least_likely_word(input_text, model, tokenizer)


def predict_coherent_least_likely_word(input_text, model, tokenizer):
    encoding = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=512, truncation=True,
                         pad_to_max_length=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        last_token_logits = logits[:, -1, :]

        # Adjust randomness with temperature
        temperature = 2
        logits_with_temperature = last_token_logits / temperature

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits_with_temperature, dim=-1)

        # Sort the probabilities to find the least likely words
        sorted_prob, sorted_indices = torch.sort(probabilities, descending=False)  # Sort in ascending order

        # Use all indices as candidate indices
        candidate_indices = sorted_indices[0, :]

        word_indices = filterWords(candidate_indices.tolist(), tokenizer)

        word = choose_word(input_text, word_indices, tokenizer)

    return word


def predict_next_word(input_text, model, tokenizer):
    encoding = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=512, truncation=True,
                         pad_to_max_length=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.size(-1) + 1,
                                 do_sample=False)

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    next_word = prediction.split()[-1]
    return next_word


def predict_least_likely_word(input_text, model, tokenizer):
    encoding = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=512, truncation=True,
                         pad_to_max_length=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        last_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(last_token_logits, dim=-1)

        min_prob, min_index = torch.min(probabilities, dim=-1)

    least_likely_word = tokenizer.decode(min_index, skip_special_tokens=True)
    return least_likely_word


def filterWords(indices, tokenizer):
    word_indices = [idx for idx in indices
                    if not tokenizer.decode(idx).startswith('##')
                    and idx not in tokenizer.all_special_ids
                    and re.match('^[a-zA-Z0-9]+$', tokenizer.decode(idx))]
    return word_indices


def choose_word(input_text, word_indices, tokenizer):
    choices = next_pos_options(input_text)

    with open('Dictionary/en_US.dic', 'r') as file:
        raw_word_set = set(word.strip().lower() for word in file)
        word_set = [word.split('/')[0].lower() for word in raw_word_set]


    if len(choices) == 0:
        return "I"

    for idx in word_indices:
        word = tokenizer.decode([idx], skip_special_tokens=True)
        wordType = nltk.pos_tag([word])[0][1]
        if wordType in choices:
            if is_valid_word(word, word_set):  # I nested these to avoid unnecessary computation
                return word.lower()

    return "I"


def next_pos_options(sentence):
    # Define possible follow-ups for each part of speech
    follow_ups = {
        'CC': ['NN', 'NNS', 'JJ', 'RB', 'PRP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Coordinating conjunction
        'CD': ['NN', 'NNS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'CC'],  # Cardinal number
        'DT': ['NN', 'NNS', 'JJ'],  # Determiner
        'EX': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Existential there
        'FW': [],  # Foreign word
        'IN': ['NN', 'NNS', 'PRP', 'JJ', 'RB', 'VBG'],  # Preposition or subordinating conjunction
        'JJ': ['NN', 'NNS', 'CC', ',', '.'],  # Adjective
        'JJR': ['NN', 'NNS'],  # Adjective, comparative
        'JJS': ['NN', 'NNS'],  # Adjective, superlative
        'LS': [],  # List item marker
        'MD': ['VB'],  # Modal
        'NN': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'NN', 'NNS', 'CC', ',', '.'],  # Noun, singular or mass
        'NNS': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'NN', 'NNS', 'CC', ',', '.'],  # Noun, plural
        'NNP': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'NN', 'NNS', 'CC', ',', '.'],  # Proper noun, singular
        'NNPS': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'NN', 'NNS', 'CC', ',', '.'],  # Proper noun, plural
        'PDT': ['NN', 'NNS', 'JJ'],  # Predeterminer
        'POS': ['NN', 'NNS'],  # Possessive ending
        'PRP': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'CC', ',', '.'],  # Personal pronoun
        'PRP$': ['NN', 'NNS'],  # Possessive pronoun
        'RB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB', 'CC', ',', '.'],  # Adverb
        'RBR': ['JJ', 'RB'],  # Adverb, comparative
        'RBS': ['JJ'],  # Adverb, superlative
        'RP': ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Particle
        'SYM': [],  # Symbol
        'TO': ['VB'],  # to
        'UH': [],  # Interjection
        'VB': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, base form
        'VBD': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, past tense
        'VBG': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, gerund/present participle
        'VBN': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, past participle
        'VBP': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, non-3rd person singular present
        'VBZ': ['NN', 'NNS', 'PRP', 'RB', 'CC', ',', '.'],  # Verb, 3rd person singular present
        'WDT': ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Wh-determiner
        'WP': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Wh-pronoun
        'WP$': ['NN', 'NNS'],  # Possessive wh-pronoun
        'WRB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Wh-adverb
        ',': ['NN', 'NNS', 'JJ', 'RB', 'PRP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Comma
        '.': []  # Period (end of sentence)
    }

    words = nltk.word_tokenize(sentence)
    tagged = pos_tag(words)

    last_word, last_pos = tagged[-1]

    return follow_ups.get(last_pos, [])


def is_valid_word(word, word_set):
    return word.lower() in word_set
