import torch
import faiss
import numpy as np

from transformers import BertTokenizer, BertModel
from datasets import load_dataset

def get_bert_vector(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    return outputs[0][0][0].reshape(1, -1).detach().numpy()  # The last hidden-state is the first element of the output 


sentence_list = ["I am a dog", "I am a pig", "I am a cat", "You are a very pig"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

index = faiss.IndexFlatL2(768)

for i, sentence in enumerate(sentence_list):
    index.add(get_bert_vector(sentence))

print(index.ntotal)

new_sentence = "You are a cat"
distances, indices = index.search(get_bert_vector(sentence), 3)

print(indices[0])

for i in indices[0]:
    print(sentence_list[i])

faiss.write_index(index, "bert_index.faiss")