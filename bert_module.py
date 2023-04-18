from transformers import BertTokenizer, BertModel
# rom sentence_transformers import SentenceTransformer, util
# import torch.nn.functional as F

import torch
# import bert


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

model = BertModel.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


def similarity(text1, text2):
    # Tokenize the input texts
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")

    # Compute the embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Calculate the cosine similarity between the embeddings
    embeddings1 = outputs1.last_hidden_state[:, 0, :]
    embeddings2 = outputs2.last_hidden_state[:, 0, :]
    cosine_similarity = cos(embeddings1, embeddings2)

    return cosine_similarity.item()