import json
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
import copy

with open('raw_sequences.json', 'r') as f:
    sequences = json.load(f)

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
processed = []

for sequence in sequences:
    processed_sequence = copy.deepcopy(sequence)

    pre_context = sequence['pre_context']
    processed_sequence['pre_context_char_length'] = len(pre_context)
    processed_sequence['pre_context_word_count'] = len(pre_context.split())
    processed_sequence['gpt_pre_context_token_count'] = len(gpt_tokenizer.tokenize(pre_context))
    #processed_sequence['llama_pre_context_token_count'] = len(llama_tokenizer.tokenize(pre_context))
    processed_sequence['mistral_pre_context_token_count'] = len(mistral_tokenizer.tokenize(pre_context))

    text = sequence['text']
    processed_sequence["text_char_length"] = len(text)
    processed_sequence["text_word_count"] = len(text.split())
    processed_sequence["gpt_text_token_count"] = len(gpt_tokenizer.tokenize(text))
    #processed_sequence["llama_text_token_count"] = len(llama_tokenizer.tokenize(text))
    processed_sequence["mistral_text_token_count"] = len(mistral_tokenizer.tokenize(text))


    post_context = sequence["post_context"]
    processed_sequence["post_context_char_length"] = len(post_context)
    processed_sequence["post_context_word_count"] = len(post_context.split())
    processed_sequence["gpt_post_context_token_count"] = len(gpt_tokenizer.tokenize(post_context))
    #processed_sequence["llama_post_context_token_count"] = len(llama_tokenizer.tokenize(post_context))
    processed_sequence["mistral_post_context_token_count"] = len(mistral_tokenizer.tokenize(post_context))

    processed.append(processed_sequence)

with open("processed_sequences.json", 'w') as f:
    json.dump(processed, f, indent=4)
