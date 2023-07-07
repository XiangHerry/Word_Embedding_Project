# Import custom modules for WordNet and BERT
from wordnet_module import disambiguate, get_related_words
from bert_module import text_similarity

# Define a word to get meanings and parents for
word = 'paper'

# Get the related words of the word using WordNet
sense = disambiguate(word, 'context')
if sense is not None:
    # Get the related words of the sense using WordNet
    related_words = get_related_words(sense, similarity_threshold=0.8)
    print(f"Related words of '{word}':")
    for related_word in related_words:
        print(f"- {related_word}")

# Define two texts to compare using cosine similarity
text1 = "software"
text2 = "program"

# Compute the cosine similarity between the two texts using BERT
similarity_score = text_similarity(text1, text2)
print(f"The similarity between '{text1}' and '{text2}' is: {similarity_score}")
