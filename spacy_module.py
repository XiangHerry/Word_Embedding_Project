import spacy


class SpacyParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, sentence):
        doc = self.nlp(sentence)
        return doc


if __name__ == "__main__":
    parser = SpacyParser()
    sentence = "The boy likes the girl next door"
    parsed_sentence = parser.parse(sentence)

    print("Original Sentence:", sentence)
    print("\nParsed Sentence:")
    for token in parsed_sentence:
        print(f"{token.text:{10}} {token.dep_:{10}} {token.head.text:{10}} {token.head.pos_:{10}}")
