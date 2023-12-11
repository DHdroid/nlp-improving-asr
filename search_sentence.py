import torch
import faiss
import numpy as np
import pandas as pd
import time
import os
import random

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


def search_random_sentence(hypotheses, references, num_examples=10):
    range_indices = len(hypotheses)
    random_indices = random.sample(range(range_indices), num_examples)
    return [(hypotheses[x], references[x]) for x in random_indices]


def search_similar_sentence(index, new_sentence, hypotheses, references, tokenizer, model, num_examples=10):
    distances, indices = index.search(get_bert_vector(tokenizer, model, new_sentence), num_examples)
    return [(hypotheses[x], references[x]) for x in indices[0]]


def main():
    file_path = './librispeech-pc/filtered_merged.csv'
    hypotheses, references = read_data_from_csv(file_path)
    
    global tokenizer, model
    tokenizer, model = get_bert_tokenizer_model()

    _ = initialize_or_load_faiss_index(hypotheses, './librispeech-pc/filtered_merged.faiss')

if __name__ == "__main__":
    main()
