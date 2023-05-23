# Import required libraries from transformers package
from transformers import BertTokenizer, BertModel
# rom sentence_transformers import SentenceTransformer, util
# import torch.nn.functional as F
from functools import lru_cache
import torch
# import bert

# Create a cosine similarity object from PyTorch
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# Load the pre-trained BERT model and tokenizer from Hugging Face Transformers
model = BertModel.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Compute the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings




def emb_similarity(embeddings1, embeddings2):
    cosine_similarity = cos(embeddings1, embeddings2)
    return cosine_similarity.item()


def text_similarity(text1, text2):
    embeddings1 = get_embeddings(text1)
    embeddings2 = get_embeddings(text2)
    cosine_similarity = cos(embeddings1, embeddings2)
    return cosine_similarity.item()



