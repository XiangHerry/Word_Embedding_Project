import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
import functools
from utilities import Singleton
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


class Concept():
    def __init__(self, pos, synset, definition, hyponyms, hyernyms):
        self.pos = pos
        self.definition = definition
        self.synset = synset
        self.hyponyms = hyponyms
        self.hypernyms = hyernyms
        self.related = []
        self.bows = []
        self.embedding = None


class SemanticNet(metaclass=Singleton):
    def __init__(self):

        self.noun_meanings = dict()
        self.verb_meanings = dict()
        self.adj_meanings = dict()
        self.adv_meanings = dict()

        # self.noun_id_concept = dict()
        # self.verb_id_concept = dict()
        # self.adj_id_concept = dict()
        # self.adv_id_concept = dict()

        self.read_synsets()
        self.compute_related()
        print('Done building network...')
        # self.compute_embeddings()

    def get_hypernyms(self, synset):
        return list({syn.name() for syn in synset.hypernyms() if syn is not None})

    def get_hyponyms(self, synset):
        return list({syn.name() for syn in synset.hyponyms() if syn is not None})

    def read_synsets(self):
        all_words = wn.words()
        print('Reading semantic network')
        count = 0
        print('=> Processing words and meanings ', end='')
        for w in all_words:
            count += 1
            if count % 10000 == 0:
                count = 0
                print('.', end='')

            nouns_list = wn.synsets(w, pos=wn.NOUN)
            verbs_list = wn.synsets(w, pos=wn.VERB)
            adjs_list = wn.synsets(w, pos=wn.ADJ)
            advs_list = wn.synsets(w, pos=wn.ADV)

            if len(nouns_list) > 0:
                self.noun_meanings[w] = []
            for syn in nouns_list:
                c = Concept('NOUN', syn, syn.definition(), self.get_hypernyms(syn), self.get_hyponyms(syn))
                self.noun_meanings[w].append(c)

            if len(adjs_list) > 0:
                self.adj_meanings[w] = []
            for syn in adjs_list:
                c = Concept('ADJ', syn, syn.definition(), self.get_hypernyms(syn), self.get_hyponyms(syn))
                self.adj_meanings[w].append(c)

            if len(verbs_list) > 0:
                self.verb_meanings[w] = []
            for syn in verbs_list:
                c = Concept('VERB', syn, syn.definition(), self.get_hypernyms(syn), self.get_hyponyms(syn))
                self.verb_meanings[w].append(c)

            if len(advs_list) > 0:
                self.adv_meanings[w] = []
            for syn in advs_list:
                c = Concept('ADV', syn, syn.definition(), self.get_hypernyms(syn), self.get_hyponyms(syn))
                self.adv_meanings[w].append(c)
        print()

    def antonyms_for_synset(self, synset):
        res = []
        append1 = res.append
        for lem in synset.lemmas():
            for related_form in lem.antonyms():
                rsynset = related_form.synset()
                append1(rsynset)
        fres = [syn.name() for syn in res]
        return fres

    def get_all_related(self, synset):
        related = []
        rs1 = set(self.antonyms_for_synset(synset))
        rs2 = {syn.name() for syn in synset.usage_domains() if syn is not None}
        rs3 = {syn.name() for syn in synset.topic_domains() if syn is not None}
        rs4 = {syn.name() for syn in synset.attributes() if syn is not None}
        rs5 = {syn.name() for syn in (synset.similar_tos() + synset.also_sees()) if syn is not None}
        rs6 = {syn.name() for syn in synset.member_holonyms() if syn is not None}
        rs7 = {syn.name() for syn in synset.part_holonyms() if syn is not None}
        rs8 = {syn.name() for syn in synset.substance_holonyms() if syn is not None}
        rs9 = {syn.name() for syn in synset.member_meronyms() if syn is not None}
        rs10 = {syn.name() for syn in synset.part_meronyms() if syn is not None}
        rs11 = {syn.name() for syn in synset.substance_meronyms() if syn is not None}
        for r in rs1:
            related.append(wn.synset(r))
        for r in rs2:
            related.append(wn.synset(r))
        for r in rs3:
            related.append(wn.synset(r))
        for r in rs4:
            related.append(wn.synset(r))
        for r in rs5:
            related.append(wn.synset(r))
        for r in rs6:
            related.append(wn.synset(r))
        for r in rs7:
            related.append(wn.synset(r))
        for r in rs8:
            related.append(wn.synset(r))
        for r in rs9:
            related.append(wn.synset(r))
        for r in rs10:
            related.append(wn.synset(r))
        for r in rs11:
            related.append(wn.synset(r))
        return related

    def compute_related(self):
        for w in self.noun_meanings:
            for m in self.noun_meanings[w]:
                m.related = self.get_coordinates(m.synset)
                m_related = self.get_all_related(m.synset)
                for mr in m_related:
                    m.related.append(mr)
                for lem in m.synset.lemma_names():
                    m.bows.append(lem.replace('_', ' '))
                for syn in m.related:
                    for lem in syn.lemma_names():
                        m.bows.append(lem.replace('_', ' '))
                m.bows = list(set(m.bows))
        for w in self.adj_meanings:
            for m in self.adj_meanings[w]:
                m.related = self.get_coordinates(m.synset)
                m.related = self.get_coordinates(m.synset)
                m_related = self.get_all_related(m.synset)
                for mr in m_related:
                    m.related.append(mr)
                for lem in m.synset.lemma_names():
                    m.bows.append(lem.replace('_', ' '))
                for syn in m.related:
                    for lem in syn.lemma_names():
                        m.bows.append(lem.replace('_', ' '))
                m.bows = list(set(m.bows))
        for w in self.verb_meanings:
            for m in self.verb_meanings[w]:
                m.coordinates = self.get_coordinates(m.synset)
                m.related = self.get_coordinates(m.synset)
                m_related = self.get_all_related(m.synset)
                for mr in m_related:
                    m.related.append(mr)
                for lem in m.synset.lemma_names():
                    m.bows.append(lem.replace('_', ' '))
                for syn in m.related:
                    for lem in syn.lemma_names():
                        m.bows.append(lem.replace('_', ' '))
                m.bows = list(set(m.bows))
        for w in self.adv_meanings:
            for m in self.adv_meanings[w]:
                m.coordinates = self.get_coordinates(m.synset)
                m.related = self.get_coordinates(m.synset)
                m_related = self.get_all_related(m.synset)
                for mr in m_related:
                    m.related.append(mr)
                for lem in m.synset.lemma_names():
                    m.bows.append(lem.replace('_', ' '))
                for syn in m.related:
                    for lem in syn.lemma_names():
                        m.bows.append(lem.replace('_', ' '))
                m.bows = list(set(m.bows))

    # this function get all hyponyms of hypernyms of one synset.
    def get_coordinates(self, synset):
        coordinates = []
        hypernyms = synset.hypernyms()
        for hypern in hypernyms:
            hyponyms = hypern.hyponyms()
            for hyponym in hyponyms:
                coordinates.append(hyponym)
        return coordinates





# def get_bows(self, synset, similarity_threshold=0.7):
#     defn = synset.definition()
#     related_words = [tok for tok in [w.lower() for w in defn.split(' ')] if tok not in stopWords and tok in semNet.noun_meanings]
#     context_embedding = get_embeddings(defn)
#     related_synsets = (synset.hypernyms() + synset.hyponyms() + self.get_coordinates(synset))
#     # all_defs_ = [(remove_special_chars(rsyn.definition())).split() for rsyn in related_synsets]
#     for rsyn in related_synsets:
#         sim = cosine_similarity(get_embeddings(rsyn.definition()).reshape(1, -1), context_embedding.reshape(1, -1))[0][0]
#         if sim >= similarity_threshold:
#             for lem in rsyn.lemma_names():
#                 if lem not in related_words:
#                     related_words.append(lem)
#     return related_words
#
#
# # this function tries to adjust the threshold so that teh result from get related words won't too many or too few.
# def adjust_threshold(num_results, iteration, max_iterations=10):
#     if iteration >= max_iterations:
#         return None
#     elif num_results > 10:
#         return 0.85
#     elif num_results < 5:
#         return 0.65
#     else:
#         return None
#
#
# def get_related_words(synset, similarity_threshold=0.7, iteration=0, max_iterations=10):
#     related_words = []
#     # Add synonyms to the list of related words
#     synonyms = synset.lemma_names()
#     for synonym in synonyms:
#         synonym = synonym.replace('_', ' ')
#         if synonym not in related_words:
#             related_words.append(synonym)
#     # Get the definition of synset and computes their embeddings.
#     context_embedding = get_embeddings(synset.definition())
#
#     # Expand the range of related_synsets
#     related_synsets = (synset.hypernyms() + synset.hyponyms() + synset.part_meronyms() + synset.part_holonyms() +
#                        get_coordinates(synset) + synset.attributes() + synset.entailments() + synset.similar_tos() +
#                        synset.also_sees() + synset.causes() + synset.member_holonyms() + synset.substance_holonyms() +
#                        synset.member_meronyms() + synset.substance_meronyms())
#     # All the related words for synset.
#     all_defs_ = [(remove_special_chars(rsyn.definition())).split() for rsyn in related_synsets]
#     for rsyn in related_synsets:
#         # similarity between the rsyn definition and original word definition.
#         sim = cosine_similarity(get_embeddings(rsyn.definition()).reshape(1, -1), context_embedding.reshape(1, -1))[0][0]
#         if sim > similarity_threshold:
#             related_syn_token = [lem.replace('_', ' ') for lem in rsyn.lemma_names()]
#             for rtoken in related_syn_token:
#                 if rtoken not in related_words:
#                     related_words.append(rtoken)
#                 if len(related_words) >= 10:
#                     return related_words
#     def_tokens = [w for w in functools.reduce(lambda a, b: a + b, all_defs_) if w not in stopWords and len(w) > 2]
#     for def_token in def_tokens:
#         if def_token not in related_words:
#             def_token_embedding = get_embeddings(def_token)
#             related_syn_token_embedding = get_embeddings(rtoken)
#             sim = cosine_similarity(def_token_embedding, related_syn_token_embedding)
#             sim = sim.item()  # Convert the similarity value to a scalar
#             if sim > similarity_threshold:
#                 related_words.append(def_token)
#                 if len(related_words) >= 10:
#                     return related_words
#     # If we haven't returned yet, it means we have less than 5 related words.
#     # We adjust the threshold and try again, unless we have already tried too many times.
#     if len(related_words) < 5 and iteration < max_iterations:
#         new_threshold = adjust_threshold(len(related_words), iteration)
#         return get_related_words(synset, new_threshold, iteration + 1)
#     else:
#         return related_words
#
#
# # # 1. combine these two methods, get average scores and do few texts: context has one/two or more words.
# def disambiguate_path_similarity(word, context):
#     # get all senses of word.
#     word_senses = wn.synsets(word)
#     context_words = context.split(', ')
#
#     best_sense = None
#     max_similarity = -1
#
#     for sense in word_senses:
#         total_similarity = 0
#         # go through every word in context words.
#         for context_word in context_words:
#             # get the senses of the context words.
#             context_senses = wn.synsets(context_word)
#             # calculate the word sense and context word sense similarity.
#             max_word_similarity = max([sense.path_similarity(context_sense) for context_sense in context_senses if sense.path_similarity(context_sense) is not None], default=0)
#             # add them to total similarity.
#             total_similarity += max_word_similarity
#
#         if total_similarity > max_similarity:
#             max_similarity = total_similarity
#             best_sense = sense
#
#     return best_sense
#
#
# def disambiguate_bert_embeddings(word, context):
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
#
#
# def combined_disambiguate(word, context, weight=0.5):
#     # weight determines the contribution of each method, must be between 0 and 1.
#     assert 0 <= weight <= 1, "Weight must be between 0 and 1."
#
#     # disambiguation using path_similarity method
#     synset_1 = disambiguate_path_similarity(word, context)
#     # disambiguation using BERT embeddings method
#     synset_2 = disambiguate_bert_embeddings(word, context)
#
#     # get BERT embeddings for definitions of each synset
#     emb_1 = get_embeddings(synset_1.definition())
#     emb_2 = get_embeddings(synset_2.definition())
#
#     # get BERT embeddings for context
#     context_emb = get_embeddings(context)
#
#     # calculate cosine similarity between context and each synset
#     sim_1 = cosine_similarity(context_emb.reshape(1, -1), emb_1.reshape(1, -1))[0][0]
#     sim_2 = cosine_similarity(context_emb.reshape(1, -1), emb_2.reshape(1, -1))[0][0]
#
#     # combine the results with a weighted average
#     if weight * sim_1 + (1 - weight) * sim_2 > 0.5:
#         return synset_1
#     else:
#         return synset_2
#
#
# def try_get_related_words():
#     context = 'a written work published in a print or electronic medium'
#     syn = combined_disambiguate('paper', context, weight=0.5)  # Use combined_disambiguate instead of disambiguate
#     print(syn)
#     bow = get_related_words(syn, 0.80)
#     for w in bow:
#         print(w)
#
#
# # try_get_related_words()


semNet = SemanticNet()


def create_context_vector(tokens):
    v = sum([get_embeddings(tok) for tok in tokens])
    # sz = len(tokens)
    # if sz == 0:
    #     return 0
    # weight = 1.0 / sz
    # v = weight * get_embeddings(tokens[0])
    # k = 1
    # while k < sz:
    #     # v = v + weight * get_embeddings(tokens[k])
    #     v = v + get_embeddings(tokens[k])
    #     k = k + 1
    return v


def show_all_meanings(contextVector):
    while True:
        token = input('Enter word: ')
        ms = semNet.noun_meanings[token]
        context_vector = create_context_vector(contextVector)
        mx = 0
        dfn = ''
        for m in ms:
            sim = emb_similarity(sum(get_embeddings(m.bows)), context_vector)
            if sim > mx:
                mx = sim
                dfn = m.definition
        print('=> ' + dfn)


show_all_meanings(['program', 'computer', 'software', 'programming'])
# def try_get_bow():
#     while True:
#         word_meaning_num = (input('Enter word and meaning number: ')).split(' ')
#         word = word_meaning_num[0]
#         mnum = int(word_meaning_num[1])
#         synsets = wn.synsets(word)
#         # bow = get_related_words(synsets[mnum - 1], 0.80)
#         bow = semNet.get_bows(synsets[mnum - 1], 0.80)
#         for w in bow:
#             print(w)
#
# try_get_bow()


