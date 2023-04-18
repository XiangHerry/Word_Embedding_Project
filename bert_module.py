# Import required libraries from transformers package
from transformers import BertTokenizer, BertModel
# rom sentence_transformers import SentenceTransformer, util
# import torch.nn.functional as F

import torch
# import bert

# Create a cosine similarity object from PyTorch
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# Load the pre-trained BERT model and tokenizer from Hugging Face Transformers
model = BertModel.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Define a function to compute the cosine similarity between two texts
def similarity(text1, text2):
    # Tokenize the input texts using the BERT tokenizer
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")

    # Compute the embeddings of the input texts using the pre-trained BERT model
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Calculate the cosine similarity between the embeddings
    embeddings1 = outputs1.last_hidden_state[:, 0, :]
    embeddings2 = outputs2.last_hidden_state[:, 0, :]
    cosine_similarity = cos(embeddings1, embeddings2)

    # Return the cosine similarity score as a float
    return cosine_similarity.item()

