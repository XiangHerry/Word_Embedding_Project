# Import WordNet corpus from NLTK library
import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from bert_module import get_embeddings, emb_similarity
from utilities import get_intersection

stopWords = list(set(stopwords.words('english')))


def remove_special_chars(txt):
    spChars = ',.<>/?`~-_()*&^%$#@][{};:\'"'
    res = ''
    for ch in txt:
        if ch not in spChars:
            res = res + str(ch)
    return res

def get_coordinates(synset):
    coordinates = []
    hypernyms = synset.hypernyms()
    for hypern in hypernyms:
        hyponyms = hypern.hyponyms()
        for hyponym in hyponyms:
            coordinates.append(hyponym)
    return coordinates

# original version of get_related words method.
# def get_related_words(synset, similarity_threshold=0.7):
#     all_text_list = [synset.definition()]
#     related_synsets = synset.hypernyms() + synset.hyponyms() + synset.part_meronyms() + synset.part_holonyms()
#     related_words = []
#     related_definitions = []
#     for rsyn in related_synsets:
#         all_text_list.append(rsyn.definition())
#         related_definitions.append(rsyn.definition())
#         all_text = ' '.join(all_text_list)
#         tokens = (remove_special_chars(all_text)).split()
#         bestSim1 = 0
#         related_words = []
#         related_syn_token = (rsyn.lemma_names()[0]).replace('_', ' ')
#         sim2 = emb_similarity(get_embeddings(rsyn.definition()), get_embeddings(synset.definition()))
#         for token in tokens:
#             if token not in stopWords and len(token) > 2:
#                 sim1 = emb_similarity(get_embeddings(token), get_embeddings(related_syn_token))
#                 sim = 0.6 * sim1 + 0.4 * sim2
#                 if sim > similarity_threshold:
#                     if token not in related_words:
#                         related_words.append(token)
#                     if related_syn_token not in related_words:
#                         related_words.append(related_syn_token)
#     return related_words
import functools

def get_related_words(synset, context, similarity_threshold=0.7):
    related_words = []
    # context_embedding = get_embeddings(context)
    related_syn_token = (synset.lemma_names()[0]).replace('_', ' ')
    context_embedding = get_embeddings(synset.definition())
    # Get the related synsets (hypernyms, hyponyms, part meronyms, part holonyms) no more other related words.
    related_synsets = (synset.hypernyms() + synset.hyponyms() + synset.part_meronyms() + synset.part_holonyms() + get_coordinates(synset))
    all_defs_ = [(remove_special_chars(rsyn.definition())).split() for rsyn in related_synsets]
    for rsyn in related_synsets:
        sim = cosine_similarity(get_embeddings(rsyn.definition()).reshape(1, -1), context_embedding.reshape(1, -1))[0][0]
        if sim > similarity_threshold:
            related_syn_token = [lem.replace('_', ' ') for lem in rsyn.lemma_names()]
            for rtoken in related_syn_token:
                related_words.append(rtoken)
    def_tokens = [w for w in functools.reduce(lambda a, b: a + b, all_defs_) if w not in stopWords and len(w) > 2]
    for def_token in def_tokens:
        sim = cosine_similarity(get_embeddings(def_token).reshape(1, -1), get_embeddings(related_syn_token).reshape(1, -1))[0][0]
        if sim > similarity_threshold:
            related_words.append(rtoken)

    # # Initialize the list of related words
    # related_words = []
    # # Iterate through the related synsets
    # for rsyn in related_synsets:
    #     # Get the first lemma name for the related synset and replace underscores with spaces
    #     related_syn_token = (rsyn.lemma_names()[0]).replace('_', ' ')
    #     # Calculate the cosine similarity between the related synset definition and context embeddings
    #     sim2 = cosine_similarity(get_embeddings(rsyn.definition()).reshape(1, -1), context_embedding.reshape(1, -1))[0][0]
    #     # Remove special characters and split the related synset definition into tokens
    #     tokens = [w for w in (remove_special_chars(rsyn.definition())).split() if w not in stopWords and len(w) > 2]
    #     # Iterate through the tokens
    #     for token in tokens:
    #         # Check if the token is not a stopword and its length is greater than 2
    #         # Calculate the cosine similarity between the token and related synset token embeddings
    #         sim1 = cosine_similarity(get_embeddings(token).reshape(1, -1),
    #                                  get_embeddings(related_syn_token).reshape(1, -1))[0][0]
    #         # Calculate the final similarity value by combining sim1 and sim2 with different weights
    #         sim = 0.3 * sim1 + 0.7 * sim2
    #         # Check if the final similarity value is greater than the similarity threshold
    #         if sim > similarity_threshold:
    #             # Add the token to the related_words list if it's not already there
    #             if token not in related_words:
    #                 related_words.append(token)
    #             # Add the related_syn_token to the related_words list if it's not already there
    #             if related_syn_token not in related_words:
    #                 related_words.append(related_syn_token)

    # Return the list of related words
    return related_words


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
# try_disambiguate()


# todo: testing Get Related Bag of Words
def try_get_related_words():
    # syn = disambiguate('check', 'examine (something) in order to determine its accuracy, quality, or condition, or to detect the presence of something.')
    # syn = disambiguate('computer', 'machine for computing and executing code and algorithms on data')
    # context = 'an event or an anniversary where many people gather for dancing, joking, laughing and socializing'
    # context = 'related to politics, as a political organization that advocates a certain political ideology'
    context = 'apples, ornages, bananas, fruit'
    # context = 'the daily news in an object called a newspaper that usually comes out every day'
    # context = 'research publication like a journal paper or a conference paper'
    syn = disambiguate('apple', context)
    coordinates = get_coordinates(syn)
    bow = get_related_words(syn, context, similarity_threshold=0.80)
    # all_other = []
    # for coordinate in coordinates:
    #     more_words = get_related_words(coordinate, context, similarity_threshold=0.8)
    #     all_other.append(more_words)
    # intersection = get_intersection(all_other)
    # bow = list(set(bow + intersection))
    for w in bow:
        print(w)
try_get_related_words()

# def try_get_related_words():
#     # syn = disambiguate('check', 'examine (something) in order to determine its accuracy, quality, or condition, or to detect the presence of something.')
#     # syn = disambiguate('computer', 'machine for computing and executing code and algorithms on data')
#     syn = disambiguate('apple', 'the apple fruit that we eat and is sweet or the apple tree')
#     coordinates = get_coordinates(syn)
#     bow = get_related_words(syn)
#     all_other = []
#     for coordinate in coordinates:
#         more_words = get_related_words(coordinate)
#         all_other.append(more_words)
#     intersection = get_intersection(all_other)
#     bow = list(set(bow + intersection))
#     for w in bow:
#         print(w)
# try_get_related_words()



# c.parents = {syn.name() for syn in synset.hypernyms() if syn is not None}
#         c.children = {syn.name() for syn in synset.hyponyms() if syn is not None}
#         c.attributes = {syn.name() for syn in synset.attributes() if syn is not None}
#         c.entailments = {syn.name() for syn in synset.entailments() if syn is not None}
#         c.similar_to = {syn.name() for syn in (synset.similar_tos() + synset.also_sees()) if syn is not None}
#         c.causes = {syn.name() for syn in synset.causes() if syn is not None}
#         c.member_holonyms = {syn.name() for syn in synset.member_holonyms() if syn is not None}
#         c.part_holonyms = {syn.name() for syn in synset.part_holonyms() if syn is not None}
#         c.substance_holonyms = {syn.name() for syn in synset.substance_holonyms() if syn is not None}
#         c.member_meronyms = {syn.name() for syn in synset.member_meronyms() if syn is not None}
#         c.part_meronyms = {syn.name() for syn in synset.part_meronyms() if syn is not None}
#         c.substance_meronyms = {syn.name() for syn in synset.substance_meronyms() if syn is not None}
#         c.related_verbs = set(c.related_cat_synsets(synset, 'v'))
#         c.related_nouns = set(c.related_cat_synsets(synset, 'n'))
#         c.related_adjs = set(c.related_cat_synsets(synset, 'a'))
#         c.related_advs = set(c.related_cat_synsets(synset, 'r'))
#         c.opposites = set(c.antonyms_for_synset(synset))
#         c.usage_domains = {syn.name() for syn in synset.usage_domains() if syn is not None}
#         c.topic_domains = {syn.name() for syn in synset.topic_domains() if syn is not None}
#         c.region_domains = {syn.name() for syn in synset.region_domains() if syn is not None}