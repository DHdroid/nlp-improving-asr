import torch
import faiss
import numpy as np
import pandas as pd
import time
import os

from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from generate_prompt import generate_gpt2_prompt


def get_bert_tokenizer_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print('Model is loaded')
    return tokenizer, model

def read_data_from_csv(file_path):
    print(file_path)
    data = pd.read_csv(file_path)
    hypotheses = data['hypothesis'].tolist()
    references = data['reference'].tolist()
    return hypotheses, references

def get_bert_vector(tokenizer, model, sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    return outputs[0][0][0].reshape(1, -1).detach().numpy()

def initialize_or_load_faiss_index(sentence_list, index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(768)
        for i, sentence in enumerate(sentence_list):
            index.add(get_bert_vector(tokenizer, model, sentence))
        faiss.write_index(index, index_path)
    return index


def search_similar_sentences(file_path, index_path, texts, gpt2_tokenizer):
    hypotheses, references = read_data_from_csv(file_path)
    
    global tokenizer, model  # Make tokenizer and model accessible globally
    tokenizer, model = get_bert_tokenizer_model()

    index = initialize_or_load_faiss_index(hypotheses, './bert_index.faiss')

    ret_texts = []
    ret_tokens = []
    # print(texts)
    for t in texts:
        new_sentence = t
        d = search_similar_sentence(index, new_sentence, hypotheses, references)
        p = generate_gpt2_prompt(d, new_sentence, gpt2_tokenizer)
        ret_texts.append(p + t)
        t = gpt2_tokenizer.encode(p + t)
        # tokens = gpt2_tokenizer.tokenize(p + t)
        # t = gpt2_tokenizer.convert_tokens_to_ids(tokens)
        ret_tokens.append(t)
    return ret_tokens

#use this
def search_similar_sentence(index, new_sentence, hypotheses, references, tokenizer, model, num_examples=3):
    distances, indices = index.search(get_bert_vector(tokenizer, model, new_sentence), num_examples)
    # breakpoint()
    return [(hypotheses[x], references[x]) for x in indices[0]]


def main():
    file_path = './base_dev_wrong.csv'  # Replace 'path_to_your_csv_file.csv' with your file path
    hypotheses, references = read_data_from_csv(file_path)
    
    global tokenizer, model  # Make tokenizer and model accessible globally
    tokenizer, model = get_bert_tokenizer_model()

    index = initialize_or_load_faiss_index(hypotheses, './bert_index.faiss')
    # initialize_or_load_faiss_index(references, './bert_refer.faiss')

    # new_sentence = "Mr. Quilter is the apostle of the middle classes."
    # search_similar_sentences(index, new_sentence, hypotheses, references)

if __name__ == "__main__":
    main()
