import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
import functools
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from bert_module import get_embeddings, emb_similarity
from utilities import get_intersection
import torch
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
stopWords = list(set(stopwords.words('english')))


def remove_special_chars(txt):
    spChars = ',.<>/?`~-_()*&^%$#@][{};:\'"'
    res = ''
    for ch in txt:
        if ch not in spChars:
            res = res + str(ch)
    return res


# this function get all hypernyms and hyponyms of one synset.
def get_coordinates(synset):
    coordinates = []
    hypernyms = synset.hypernyms()
    for hypern in hypernyms:
        hyponyms = hypern.hyponyms()
        for hyponym in hyponyms:
            coordinates.append(hyponym)
    return coordinates

# this function tries to adjust the threshold so that teh result from get related words won't too many or too few.
def adjust_threshold(num_results, iteration, max_iterations=10):
    if iteration >= max_iterations:
        return None
    elif num_results > 10:
        return 0.85
    elif num_results < 5:
        return 0.65
    else:
        return None


def get_related_words(synset, similarity_threshold=0.7, iteration=0, max_iterations=10):
    related_words = []
    # Add synonyms to the list of related words
    synonyms = synset.lemma_names()
    for synonym in synonyms:
        synonym = synonym.replace('_', ' ')
        if synonym not in related_words:
            related_words.append(synonym)
    # Get the definition of synset and computes their embeddings.
    context_embedding = get_embeddings(synset.definition())

    # Expand the range of related_synsets
    related_synsets = (synset.hypernyms() + synset.hyponyms() + synset.part_meronyms() + synset.part_holonyms() +
                       get_coordinates(synset) + synset.attributes() + synset.entailments() + synset.similar_tos() +
                       synset.also_sees() + synset.causes() + synset.member_holonyms() + synset.substance_holonyms() +
                       synset.member_meronyms() + synset.substance_meronyms())
    # All the related words for synset.
    all_defs_ = [(remove_special_chars(rsyn.definition())).split() for rsyn in related_synsets]
    for rsyn in related_synsets:
        # similarity between the rsyn definition and original word definition.
        sim = cosine_similarity(get_embeddings(rsyn.definition()).reshape(1, -1), context_embedding.reshape(1, -1))[0][
            0]
        if sim > similarity_threshold:
            related_syn_token = [lem.replace('_', ' ') for lem in rsyn.lemma_names()]
            for rtoken in related_syn_token:
                if rtoken not in related_words:
                    related_words.append(rtoken)
                if len(related_words) >= 10:
                    return related_words
    def_tokens = [w for w in functools.reduce(lambda a, b: a + b, all_defs_) if w not in stopWords and len(w) > 2]
    for def_token in def_tokens:
        if def_token not in related_words:
            def_token_embedding = get_embeddings(def_token)
            related_syn_token_embedding = get_embeddings(rtoken)
            sim = cosine_similarity(def_token_embedding, related_syn_token_embedding)
            sim = sim.item()  # Convert the similarity value to a scalar
            if sim > similarity_threshold:
                related_words.append(def_token)
                if len(related_words) >= 10:
                    return related_words
    # If we haven't returned yet, it means we have less than 5 related words.
    # We adjust the threshold and try again, unless we have already tried too many times.
    if len(related_words) < 5 and iteration < max_iterations:
        new_threshold = adjust_threshold(len(related_words), iteration)
        return get_related_words(synset, new_threshold, iteration + 1)
    else:
        return related_words



def disambiguate(word, context):
    # get all senses of word.
    word_senses = wn.synsets(word)
    context_words = context.split(', ')

    best_sense = None
    max_similarity = -1

    for sense in word_senses:
        total_similarity = 0
        # go through every word in context words.
        for context_word in context_words:
            # get the senses of the context words.
            context_senses = wn.synsets(context_word)
            # calculate the word sense and context word sense similarity.
            max_word_similarity = max([sense.path_similarity(context_sense) for context_sense in context_senses if sense.path_similarity(context_sense) is not None], default=0)
            # add them to total similarity.
            total_similarity += max_word_similarity

        if total_similarity > max_similarity:
            max_similarity = total_similarity
            best_sense = sense

    return best_sense


# def disambiguate(word, context):
#     context_embedding = get_embeddings(context)
#     synsets = wn.synsets(word)
#
#     definitions = []
#     for synset in synsets:
#         definitions.append(synset.definition())
#     mx = 0
#     idx = 0
#     sz = len(definitions)
#     k = 0
#     while k < sz:
#         definition_embedding = get_embeddings(definitions[k])
#         sim = emb_similarity(definition_embedding, context_embedding)
#         if sim > mx:
#             mx = sim
#             idx = k
#         k = k + 1
#     return synsets[idx]


def try_get_related_words():
    context = 'apples, ornages, bananas, fruit'
    syn = disambiguate('apple', context)
    print(syn)
    bow = get_related_words(syn, 0.80)
    for w in bow:
        print(w)



try_get_related_words()
