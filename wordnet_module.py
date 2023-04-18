from nltk.corpus import wordnet as wn


def get_meanings(word):
    synsets = wn.synsets(word)
    meanings = []
    for synset in synsets:
        meanings.append(synset.definition())
    return meanings


def get_parent(word):
    synsets = wn.synsets(word)
    parents = []
    for synset in synsets:
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            parents.append(hypernym.lemma_names())
    return parents