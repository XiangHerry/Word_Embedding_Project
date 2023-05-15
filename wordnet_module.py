# Import WordNet corpus from NLTK library
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from bert_module import get_embeddings, emb_similarity
from utilities import get_intersection

stopWords = list(set(stopwords.words('english')))


class WordNet():
    def remove_special_chars(txt):
        spChars = ',.<>/?`~-_()*&^%$#@][{};:\'"'
        res = ''
        for ch in txt:
            if ch not in spChars:
                res = res + str(ch)
        return res

    # Get all meanings of a word
    def get_meanings(word):
        synsets = wn.synsets(word)  # Get all Synset objects related to the word
        meanings = []
        for synset in synsets:
            meanings.append(synset.definition())  # Add the definition of each Synset to the list of meanings
        return meanings

    def get_coordinates(synset):
        coordinates = []
        hypernyms = synset.hypernyms()
        for hypern in hypernyms:
            hyponyms = hypern.hyponyms()
            for hyponym in hyponyms:
                coordinates.append(hyponym)
        return coordinates

    def get_parent(synset):
        parents = []
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            parents.append(hypernym.lemma_names())
        return parents

    # Get all parent synsets of a word
    def get_parents_of_all_meanings(word):
        synsets = wn.synsets(word)  # Get all Synset objects related to the word
        parents = []
        for synset in synsets:
            hypernyms = synset.hypernyms()  # Get all hypernym Synset objects of the Synset
            for hypernym in hypernyms:
                parents.append(hypernym.lemma_names())
        return parents

    # given some meaning, get a bag of related words
    def get_related_words(synset):
        all_text_list = [synset.definition()]
        hypernyms = synset.hypernyms()
        for hypern in hypernyms:
            all_text_list.append(hypern.definition())
        all_text = ' '.join(all_text_list)
        tokens = (remove_special_chars(all_text)).split()
        bow = [w for w in tokens if w not in stopWords and len(w) > 2]
        return bow

    def disambiguate(word, context):
        context_embedding = get_embeddings(context)
        synsets = wn.synsets(word)
        definitions = []
        for synset in synsets:
            definitions.append(synset.definition())
        mx = 0
        idx = 0
        sz = len(definitions)
        k = 0
        while k < sz:
            definition_embedding = get_embeddings(definitions[k])
            sim = emb_similarity(definition_embedding, context_embedding)
            if sim > mx:
                mx = sim
                idx = k
            k = k + 1
        return synsets[idx]





# todo: testing Disambiguation : data ::= [(word, context)]
def try_disambiguate():
    test_disambiguation = [('paper', 'research publication like a journal paper or a conference paper'),
                           ('paper', 'the daily news in an object called a newspaper that usually comes out every day'),
                           ('party', 'an event or an anniversary where many people gather for dancing, joking, laughing and socializing'),
                           ('party', 'related to politics, as a political organization that advocates a certain political ideology'),
                           ('tank', 'an armored vehicle that usually has a cannon and is ued in military operations'),
                           ('tank', 'a vessel that is used to hold liquid in it'),
                           ]

    for pair in test_disambiguation:
        word = pair[0]
        context = pair[1]
        syn = disambiguate(word, context)
        print('disambiguate[' + word + ']' + '\n'
              + 'CONTEXT = ' + context + '\n'
              + 'SELECTED MEANING = ' + syn.definition() + '\n')
try_disambiguate()


# todo: testing Get Related Bag of Words
def try_get_related_words():
    syn = disambiguate('program', 'an algorithm, or program or code that is implemented in some programming language to run its code on a computer')
    coordinates = get_coordinates(syn)
    bow = get_related_words(syn)
    all_other = []
    for coordinate in coordinates:
        more_words = get_related_words(coordinate)
        all_other.append(more_words)
    intersection = get_intersection(all_other)
    bow = list(set(bow + intersection))
    for w in bow:
        print(w)
# try_get_related_words()



