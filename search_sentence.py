import torch
import faiss
import numpy as np
import pandas as pd
import time
import os
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

def get_bert_tokenizer_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print('Model is loaded')
    return tokenizer, model

def read_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    hypotheses = data['hypothesis'].tolist()
    references = data['reference'].tolist()
    return hypotheses, references

def get_bert_vector(tokenizer, model, sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    return outputs[0][0][0].reshape(1, -1).detach().numpy()

def initialize_or_load_faiss_index(sentence_list):
    if os.path.exists('./bert_index.faiss'):
        index = faiss.read_index('./bert_index.faiss')
    else:
        index = faiss.IndexFlatL2(768)
        for i, sentence in enumerate(sentence_list):
            index.add(get_bert_vector(tokenizer, model, sentence))
        faiss.write_index(index, "bert_index.faiss")
    return index


def search_similar_sentences(index, new_sentence, sentence_list):
    start = time.time()
    distances, indices = index.search(get_bert_vector(tokenizer, model, new_sentence), 3)
    print(f'The distances are calculated. Indices sequences are {indices[0]}. The elapsed time is {time.time() - start} sec')
    print([sentence_list[x] for x in indices[0]])


#use this
def search_similar_sentences_directly(index_path, new_sentence, beam_size):
    index = faiss.read_index(index_path)
    distances, indices = index.search(get_bert_vector(tokenizer, model, new_sentence), beam_size)
    return [sentence_list[x] for x in indices[0]]


def main():
    file_path = './TalkFile_test.csv'  # Replace 'path_to_your_csv_file.csv' with your file path
    hypotheses, _ = read_data_from_csv(file_path)
    
    global tokenizer, model  # Make tokenizer and model accessible globally
    tokenizer, model = get_bert_tokenizer_model()

    index = initialize_or_load_faiss_index(hypotheses)

    new_sentence = "Mr. Quilter is the apostle of the middle classes."
    search_similar_sentences(index, new_sentence, hypotheses)

if __name__ == "__main__":
    main()
