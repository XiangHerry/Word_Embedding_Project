# Import custom modules for WordNet and BERT
import wordnet_module
from bert_module import similarity

# Define a word to get meanings and parents for
word = 'paper'

# Get the meanings of the word using WordNet
meanings = wordnet_module.get_meanings(word)
print(f"Meanings of '{word}':")
for meaning in meanings:
    print(f"- {meaning}")

# Get the parents of the word using WordNet
parents = wordnet_module.get_parent(word)
print(f"\nParents of '{word}':")
for parent in parents:
    print(f"- {', '.join(parent)}")

# Define two texts to compare using cosine similarity
text1 = "software"
text2 = "program"

# Compute the cosine similarity between the two texts using BERT
similarity_score = similarity(text1, text2)
print(f"The similarity between '{text1}' and '{text2}' is: {similarity_score}") 
