# Import WordNet corpus from NLTK library
from nltk.corpus import wordnet as wn


# Get all meanings of a word
def get_meanings(word):
    synsets = wn.synsets(word)  # Get all Synset objects related to the word
    meanings = []
    for synset in synsets:
        meanings.append(synset.definition())  # Add the definition of each Synset to the list of meanings
    return meanings


# Get all parent synsets of a word
def get_parent(word):
    synsets = wn.synsets(word)  # Get all Synset objects related to the word
    parents = []
    for synset in synsets:
        hypernyms = synset.hypernyms()  # Get all hypernym Synset objects of the Synset
        for hypernym in hypernyms:
            parents.append(hypernym.lemma_names())  # Add the lemma names of each hypernym Synset to the list of parents
    return parents
