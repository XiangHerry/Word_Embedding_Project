# Import Spacy library
import spacy


# Define a class for parsing text using Spacy
class SpacyParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Load the English language model in Spacy

    def parse(self, sentence):
        doc = self.nlp(sentence)  # Parse the input sentence using Spacy
        return doc


if __name__ == "__main__":
    parser = SpacyParser()  # Create a new instance of SpacyParser
    sentence = "The boy likes the girl next door"  # Define a sentence to parse
    parsed_sentence = parser.parse(sentence)  # Parse the sentence using SpacyParser
    print("Original Sentence:", sentence)
    print("\nParsed Sentence:")
    for token in parsed_sentence:  # Iterate over each token in the parsed sentence
        print(f"{token.text:{10}} {token.dep_:{10}} {token.head.text:{10}} {token.head.pos_:{10}}")
        # Print the text, dependency label, head text, and head part-of-speech tag of each token
