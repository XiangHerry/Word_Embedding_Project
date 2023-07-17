# Import required libraries from transformers package
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
# from functools import lru_cache
import torch
# import bert
from collections import Counter

# Create a cosine similarity object from PyTorch
# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
cos = torch.nn.CosineSimilarity(dim=0)

# Load the pre-trained BERT model and tokenizer from Hugging Face Transformers
model = BertModel.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


model2 = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(text):
    emb2 = model2.encode(text, convert_to_tensor=True)
    return emb2



def emb_similarity(embeddings1, embeddings2):
    cosine_similarity = cos(embeddings1, embeddings2)
    return cosine_similarity.item()


def text_similarity(text1, text2):
    embeddings1 = get_embeddings(text1)
    embeddings2 = get_embeddings(text2)
    cosine_similarity = cos(embeddings1, embeddings2)
    return cosine_similarity.item()









